from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from models.dynamic_router import DynamicEvidenceSMSModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class SMSSpamDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        encoding = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                int(self.labels[index]),
                dtype=torch.long
            )
        }


# --------------------------------------------------
# Load test data
# --------------------------------------------------

test_df = pd.read_csv(
    PROJECT_DIR
    / "data"
    / "processed"
    / "test.csv"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

test_dataset = SMSSpamDataset(
    test_df,
    tokenizer,
    MAX_LENGTH
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

print("\nLoading trained checkpoint...")

model = DynamicEvidenceSMSModel(
    model_name=MODEL_NAME,
    num_classes=2
)

checkpoint_path = (
    PROJECT_DIR
    / "models"
    / "saved"
    / "dynamic_evidence_model.pt"
)

state_dict = torch.load(
    checkpoint_path,
    map_location=DEVICE
)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

print("Checkpoint loaded successfully.")


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_mode(mode):

    predictions = []
    true_labels = []

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs["logits"]

            routing_weights = outputs[
                "routing_weights"
            ]

            # --------------------------------------------------
            # Get the three expert representations again
            # --------------------------------------------------

            encoder_output = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            sequence_output = encoder_output.last_hidden_state

            semantic = model.semantic_expert(
                sequence_output
            )

            pattern = model.pattern_expert(
                sequence_output
            )

            entity_context = model.entity_context_expert(
                sequence_output,
                attention_mask
            )

            # --------------------------------------------------
            # Ablation modifications
            # --------------------------------------------------

            if mode == "full":

                fused = (
                    semantic * routing_weights[:, 0:1]
                    + pattern * routing_weights[:, 1:2]
                    + entity_context * routing_weights[:, 2:3]
                )

            elif mode == "no_semantic":

                fused = (
                    pattern * routing_weights[:, 1:2]
                    + entity_context * routing_weights[:, 2:3]
                )

            elif mode == "no_pattern":

                fused = (
                    semantic * routing_weights[:, 0:1]
                    + entity_context * routing_weights[:, 2:3]
                )

            elif mode == "no_entity":

                fused = (
                    semantic * routing_weights[:, 0:1]
                    + pattern * routing_weights[:, 1:2]
                )

            elif mode == "static_fusion":

                fused = (
                    semantic
                    + pattern
                    + entity_context
                ) / 3.0

            else:

                raise ValueError(
                    f"Unknown mode: {mode}"
                )

            # --------------------------------------------------
            # Classification
            # --------------------------------------------------

            ablation_logits = model.classifier(
                fused
            )

            preds = torch.argmax(
                ablation_logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

    return {
        "accuracy": accuracy_score(
            true_labels,
            predictions
        ),
        "precision": precision_score(
            true_labels,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            true_labels,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            true_labels,
            predictions,
            zero_division=0
        )
    }


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

experiments = [
    ("Full Dynamic Router", "full"),
    ("Without Semantic Expert", "no_semantic"),
    ("Without Pattern Expert", "no_pattern"),
    ("Without Entity/Context Expert", "no_entity"),
    ("Static Equal Fusion", "static_fusion")
]

results = []

print("\n")
print("=" * 65)
print("CONTROLLED ABLATION STUDY")
print("=" * 65)

for name, mode in experiments:

    print(f"\nRunning: {name}")

    metrics = evaluate_mode(mode)

    result = {
        "Experiment": name,
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"]
    }

    results.append(result)

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1       : {metrics['f1']:.4f}"
    )


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_file = (
    PROJECT_DIR
    / "evaluation"
    / "ablation_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\n")
print("=" * 65)
print("FINAL ABLATION RESULTS")
print("=" * 65)

print(
    results_df.to_string(index=False)
)

print(
    f"\nResults saved to:\n{output_file}"
)
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Project path
# --------------------------------------------------

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
# Dataset class
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

        text = str(self.texts[index])
        label = int(self.labels[index])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                label,
                dtype=torch.long
            )
        }


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

TEST_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "test.csv"
)

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_FILE)

print(f"Test samples: {len(test_df)}")


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

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
# Load model
# --------------------------------------------------

print("\nLoading Dynamic Evidence Routing model...")

model = DynamicEvidenceSMSModel(
    model_name=MODEL_NAME,
    num_classes=2
)

CHECKPOINT = (
    PROJECT_DIR
    / "models"
    / "saved"
    / "dynamic_evidence_model.pt"
)

state_dict = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("Trained checkpoint loaded successfully.")


# --------------------------------------------------
# Prediction
# --------------------------------------------------

predictions = []
true_labels = []
routing_weights_list = []

print("\nRunning test evaluation...")

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

        predicted_labels = torch.argmax(
            logits,
            dim=1
        )

        predictions.extend(
            predicted_labels.cpu().numpy()
        )

        true_labels.extend(
            labels.cpu().numpy()
        )

        routing_weights_list.extend(
            routing_weights.cpu().numpy()
        )


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predictions,
    zero_division=0
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n")
print("=" * 45)
print("        SPAM SMS DETECTION RESULTS")
print("=" * 45)

print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    true_labels,
    predictions
)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predictions,
        target_names=[
            "HAM",
            "SPAM"
        ],
        zero_division=0
    )
)


# --------------------------------------------------
# Dynamic routing analysis
# --------------------------------------------------

routing_tensor = torch.tensor(
    routing_weights_list
)

average_routing = routing_tensor.mean(
    dim=0
)

print("\n")
print("=" * 45)
print("      AVERAGE EVIDENCE ROUTING")
print("=" * 45)

print(
    f"Semantic       : "
    f"{average_routing[0].item():.4f}"
)

print(
    f"Pattern        : "
    f"{average_routing[1].item():.4f}"
)

print(
    f"Entity/Context : "
    f"{average_routing[2].item():.4f}"
)

print("\nEvaluation completed successfully.")
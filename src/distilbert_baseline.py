import os
import sys
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ==========================================
# LOAD DATA
# ==========================================

TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train.csv"
)

TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "test.csv"
)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training samples: {len(train_df)}")
print(f"Test samples    : {len(test_df)}")


# ==========================================
# TOKENIZER
# ==========================================

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


# ==========================================
# DATASET
# ==========================================

class SMSDataset(Dataset):

    def __init__(self, dataframe):

        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        encoded = tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        }


train_dataset = SMSDataset(train_df)
test_dataset = SMSDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# ==========================================
# DISTILBERT-ONLY MODEL
# ==========================================

print("\nLoading DistilBERT-only model...")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

model.to(DEVICE)


# ==========================================
# TRAINING
# ==========================================

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

EPOCHS = 3

print("\n" + "=" * 60)
print("TRAINING DISTILBERT-ONLY BASELINE")
print("=" * 60)

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )


# ==========================================
# EVALUATION
# ==========================================

print("\nEvaluating DistilBERT-only model...")

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


print("\n" + "=" * 60)
print("DISTILBERT-ONLY RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ==========================================
# SAVE RESULTS
# ==========================================

results = pd.DataFrame([{
    "Model": "DistilBERT Only",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1
}])

output_path = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "distilbert_baseline_results.csv"
)

results.to_csv(
    output_path,
    index=False
)

print("\nResults saved to:")
print(output_path)

print("=" * 60)
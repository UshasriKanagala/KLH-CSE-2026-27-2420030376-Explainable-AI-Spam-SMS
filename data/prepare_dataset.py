from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path(__file__).parent
RAW_FILE = DATA_DIR / "sms_data" / "SMSSpamCollection"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading SMS dataset...")

df = pd.read_csv(
    RAW_FILE,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="utf-8"
)

# --------------------------------------------------
# Convert labels
# ham  -> 0
# spam -> 1
# --------------------------------------------------

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Remove invalid rows, if any
df = df.dropna(subset=["text", "label"])

# Make sure labels are integers
df["label"] = df["label"].astype(int)

# --------------------------------------------------
# Remove duplicate SMS messages
# --------------------------------------------------

df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print(f"Total messages after cleaning: {len(df)}")

# --------------------------------------------------
# Train / validation / test split
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# --------------------------------------------------
# Save processed datasets
# --------------------------------------------------

train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
validation_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

# --------------------------------------------------
# Display information
# --------------------------------------------------

print("\nDataset preparation completed.")

print(f"Training samples:   {len(train_df)}")
print(f"Validation samples: {len(validation_df)}")
print(f"Test samples:       {len(test_df)}")

print("\nTraining distribution:")
print(train_df["label"].value_counts())

print("\nValidation distribution:")
print(validation_df["label"].value_counts())

print("\nTest distribution:")
print(test_df["label"].value_counts())

print("\nFiles saved in:")
print(PROCESSED_DIR)
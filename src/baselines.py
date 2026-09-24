import os
import sys
import torch
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)


# ==========================================
# LOAD DATA
# ==========================================

TRAIN_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "train.csv"
)

TEST_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "test.csv"
)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Dataset columns are: label, text
X_train = train_df["text"].astype(str)
y_train = train_df["label"]

X_test = test_df["text"].astype(str)
y_test = test_df["label"]


print("=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")


# ==========================================
# TF-IDF
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# EVALUATION FUNCTION
# ==========================================

results = []


def evaluate_model(name, model):

    print("\n" + "-" * 60)
    print(f"Training: {name}")
    print("-" * 60)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


# ==========================================
# BASELINE 1
# ==========================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

evaluate_model(
    "TF-IDF + Logistic Regression",
    logistic_model
)


# ==========================================
# BASELINE 2
# ==========================================

svm_model = LinearSVC(
    random_state=42
)

evaluate_model(
    "TF-IDF + Linear SVM",
    svm_model
)


# ==========================================
# SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

output_path = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "baseline_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 60)
print("BASELINE RESULTS")
print("=" * 60)

print(
    results_df.to_string(index=False)
)

print("\nResults saved to:")
print(output_path)

print("=" * 60)
import os
import sys
import torch
from transformers import DistilBertTokenizer

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(PROJECT_ROOT)

from models.dynamic_router import DynamicEvidenceSMSModel


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "saved",
    "dynamic_evidence_model.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# LOAD TOKENIZER
# ==========================================

print("Loading tokenizer...")

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("Loading trained model...")

model = DynamicEvidenceSMSModel()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print()


# ==========================================
# SMS PREDICTION
# ==========================================

def predict_sms(message):

    encoded = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Convert logits to probabilities
        probabilities = torch.softmax(
            output["logits"],
            dim=1
        )

        # 0 = HAM, 1 = SPAM
        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0, prediction
        ].item()

        # Dynamic routing weights
        routing_weights = output[
            "routing_weights"
        ][0].cpu().numpy()

    if prediction == 1:
        label = "SPAM"
    else:
        label = "HAM"

    return label, confidence, routing_weights


# ==========================================
# INTERACTIVE DETECTOR
# ==========================================

print("=" * 60)
print("       DYNAMIC EVIDENCE SMS SPAM DETECTOR")
print("=" * 60)

print("Enter an SMS message to classify it.")
print("Type 'exit' to stop.")
print()


while True:

    message = input("Enter SMS: ")

    if message.lower() == "exit":
        print("\nExiting...")
        break

    if not message.strip():
        print("Please enter a message.\n")
        continue

    label, confidence, weights = predict_sms(message)

    print()
    print("-" * 60)

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence * 100:.2f}%")

    print()
    print("Dynamic Evidence Weights:")

    print(
        f"Semantic Expert : {weights[0] * 100:.2f}%"
    )

    print(
        f"Pattern Expert  : {weights[1] * 100:.2f}%"
    )

    print(
        f"Context Expert  : {weights[2] * 100:.2f}%"
    )

    print("-" * 60)
    print()
import os
import sys
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer

# --------------------------------------------------
# PROJECT SETUP
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from models.dynamic_router import DynamicEvidenceSMSModel


app = Flask(__name__)

DEVICE = torch.device("cpu")
MODEL_NAME = "distilbert-base-uncased"

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "models",
    "saved",
    "dynamic_evidence_model.pt"
)


# --------------------------------------------------
# LOAD TOKENIZER
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

print("Loading Dynamic Evidence Routing model...")

model = DynamicEvidenceSMSModel()

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# PREDICTION API
# --------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    data = request.get_json(
        silent=True
    )

    # Check input
    if not data or "message" not in data:

        return jsonify({
            "error": "No SMS message received."
        }), 400


    message = str(
        data["message"]
    ).strip()


    if not message:

        return jsonify({
            "error": "Please enter an SMS message."
        }), 400


    # --------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------

    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )


    # IMPORTANT:
    # DynamicEvidenceSMSModel accepts only
    # input_ids and attention_mask.
    #
    # The tokenizer may also return token_type_ids.
    # We intentionally remove it.

    inputs = {
        "input_ids":
            inputs["input_ids"].to(DEVICE),

        "attention_mask":
            inputs["attention_mask"].to(DEVICE)
    }


    # --------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------

    with torch.no_grad():

        output = model(
            **inputs
        )


        # Classification probabilities

        probabilities = torch.softmax(
            output["logits"],
            dim=-1
        )[0]


        # Predicted class

        predicted_class = int(
            torch.argmax(
                probabilities
            ).item()
        )


        # Confidence

        confidence = float(
            probabilities[
                predicted_class
            ].item()
        )


        # Dynamic routing weights

        routing_weights = (
            output["routing_weights"][0]
            .detach()
            .cpu()
            .tolist()
        )


    # --------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------

    if predicted_class == 1:

        prediction = "SPAM"

    else:

        prediction = "HAM"


    # --------------------------------------------------
    # RESPONSE
    # --------------------------------------------------

    response = {

        "prediction":
            prediction,

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "semantic":
            round(
                routing_weights[0] * 100,
                2
            ),

        "pattern":
            round(
                routing_weights[1] * 100,
                2
            ),

        "context":
            round(
                routing_weights[2] * 100,
                2
            )
    }


    print()
    print("-" * 50)
    print("SMS:", message)
    print("Prediction:", prediction)
    print(
        "Confidence:",
        round(confidence * 100, 2),
        "%"
    )

    print(
        "Semantic Expert:",
        round(routing_weights[0] * 100, 2),
        "%"
    )

    print(
        "Pattern Expert:",
        round(routing_weights[1] * 100, 2),
        "%"
    )

    print(
        "Context Expert:",
        round(routing_weights[2] * 100, 2),
        "%"
    )

    print("-" * 50)


    return jsonify(
        response
    )


# --------------------------------------------------
# START WEB SERVER
# --------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("        SPAM SMS DETECTION WEB INTERFACE")
    print("=" * 60)

    print(
        "Server running at:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 60)
    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
# Explainable AI Framework for Spam SMS Detection Using Transformers

## Project Overview

This project develops an Explainable AI framework for spam SMS detection using BERT. The system is designed to automatically classify SMS messages as Spam or Ham and provide an understandable explanation for the prediction.

When a user enters an SMS message, the text is preprocessed and given to a fine-tuned BERT model. The BERT model understands the contextual relationships between words and predicts whether the message is Spam or Ham. SHAP and LIME are then used to identify the important words that influenced the prediction.

The system provides the prediction, confidence score, and human-readable explanation through a simple web-based interface.

## Team Members

- U. Jaya Sindhura - 2420030044
- K. Ushasri - 2420030376
- P. Smitha - 2420030583

## Supervisor

Dr. K. Swanthana

## Domain

Cybersecurity and Natural Language Processing (NLP)

## Objectives

- Improve SMS spam detection using contextual understanding from BERT.
- Classify SMS messages into Spam and Ham categories.
- Reduce the limitations of traditional machine learning-based spam detection.
- Use SHAP and LIME to explain individual predictions.
- Identify important words influencing the Spam/Ham classification.
- Provide a confidence score along with the prediction.
- Make the AI-based spam detection process more transparent and understandable.

## Dataset

### SMS Spam Collection Dataset

The project uses the SMS Spam Collection Dataset, which contains 5,574 SMS messages classified into two categories:

- Spam
- Ham

Dataset Source:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

The main input to the BERT model is:

X / Input:

- SMS Message / Text

The model predicts:

Y / Output:

- Spam
- Ham

The system also provides:

- Confidence Score
- Important Words
- SHAP Explanation
- LIME Explanation

## Technologies Used

### AI and Machine Learning

- BERT
- Natural Language Processing (NLP)
- SHAP
- LIME
- PyTorch
- Hugging Face Transformers
- Scikit-learn

### Backend

- Python
- Flask

### Frontend

- HTML
- CSS
- JavaScript

### Development Tools

- VS Code
- Jupyter Notebook

## System Workflow

```text
User Input SMS
       ↓
SMS Message
       ↓
Text Preprocessing
       ↓
BERT Tokenizer
       ↓
Fine-Tuned BERT Model
       ↓
Spam / Ham Prediction
       ↓
Confidence Score
       ↓
SHAP + LIME
       ↓
Important Words / Explanation
       ↓
Display Final Result
       ↓
END

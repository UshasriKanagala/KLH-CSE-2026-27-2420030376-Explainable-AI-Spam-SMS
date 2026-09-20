# SMS Spam Detection Using Dynamic Evidence Routing Transformer

## Project Overview

This project develops an SMS spam detection system using **DistilBERT and Dynamic Evidence Routing**. The system automatically classifies SMS messages as **Spam or Ham** by analyzing different types of evidence from the message.

When a user enters an SMS, the text is preprocessed and given to the **DistilBERT encoder**. The encoded information is passed to three experts: **Semantic Expert, Pattern Expert, and Context Expert**. A **Dynamic Evidence Router** assigns instance-specific weights to each expert, and their outputs are combined using **Dynamic Weighted Evidence Fusion**.

The final system predicts whether the SMS is **Spam or Ham**.

## Team Members

* U. Jaya Sindhura - 2420030044
* K. Ushasri - 2420030376
* P. Smitha - 2420030583

## Supervisor

Dr. K. Swanthana

## Domain

Cybersecurity and Natural Language Processing (NLP)

## Objectives

* Improve SMS spam detection using DistilBERT.
* Classify SMS messages into Spam and Ham categories.
* Capture semantic, pattern, and contextual evidence.
* Dynamically assign weights to different experts.
* Combine multiple evidence sources using dynamic weighted fusion.
* Improve adaptive SMS spam detection.

## Dataset

### SMS Spam Collection Dataset

The project uses the **SMS Spam Collection Dataset**, which contains **5,574 SMS messages** classified into two categories:

* Spam
* Ham

Dataset Source:

[https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

The main input to the model is:

**X / Input:**

* SMS Message / Text

The model predicts:

**Y / Output:**

* Spam
* Ham

## Technologies Used

### AI and Machine Learning

* DistilBERT
* Transformer
* Natural Language Processing (NLP)
* PyTorch
* Hugging Face Transformers
* Scikit-learn

### Data Processing

* Python
* Pandas
* NumPy

### Development Tools

* VS Code
* Jupyter Notebook
* Google Colab
* Git & GitHub

## System Workflow

```text
User Input SMS
       ↓
SMS Message
       ↓
Text Preprocessing
       ↓
DistilBERT Encoder
       ↓
Semantic Expert
Pattern Expert
Context Expert
       ↓
Dynamic Evidence Router
       ↓
Dynamic Weighted Evidence Fusion
       ↓
Spam / Ham Prediction
       ↓
Display Final Result
       ↓
END
```

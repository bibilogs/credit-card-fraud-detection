# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using supervised models. The project compares multiple models and sampling strategies for handling an imbalanced dataset.

## Dataset

The dataset is the **Credit Card Fraud Detection** provided by TensorFlow:
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

It includes the features *Time*, *Amount*, *Class* and 28 anonymized variables (V1–V28).

The target variable is imbalanced, since legitimate transactions (Class 0) make up the vast majority of the data, whereas fraudulent transactions (Class 1) account for less than 1% of all observations.

## Models

The following models are implemented:

- Logistic Regression
- Logistic Regression + SMOTE
- Logistic Regression + Random UnderSampling
- Random Forest
- Random Forest + SMOTE
- Random Forest + Random UnderSampling
- XGBoost

## Feature Engineering

The project includes a logarithmic transformation of the transaction amount:

- Amount_log = log1p(Amount)

This transformation reduces the skewness of transaction amounts while preserving zero values.

## Evaluation Metrics

Since the dataset is imbalanced, model performance is evaluated using:

- F1-score
- Precision
- Precision-Recall AUC (PR AUC)
- Recall
- Receiver Operating Characteristic AUC (ROC AUC)

## Visualizations

The project generates:

- Confusion Matrix
- Feature Importance (tree-based models)
- Precision-Recall Curve
- ROC Curve

## Running Instruction

Execute the training script:

```bash
python src/train.py
```
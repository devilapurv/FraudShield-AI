# 🛡️ FraudShield AI

### Explainable Credit Card Fraud Detection with XGBoost, SHAP & Streamlit

FraudShield AI is an interactive machine-learning application for identifying potentially fraudulent credit card transactions. Upload a CSV, review transaction-level risk scores, explore fraud alerts, and inspect feature contributions with SHAP. The dashboard uses a dark, red-accented design.

## 📸 Application Screenshots

> Screenshot filenames must match the files in your GitHub `screenshots/` folder. Rename your images or update the paths below.

### Dashboard 1
![FraudShield Dashboard 1](screenshots/Dashboard1.png)

### Dashboard 2
![FraudShield Dashboard 2](screenshots/Dashboard2.png)

### Dashboard 3
![FraudShield Dashboard 3](screenshots/Dashboard3.png)

### Dashboard 4
![FraudShield Dashboard 4](screenshots/Dashboard4.png)

### Dashboard 5
![FraudShield Dashboard 5](screenshots/Dashboard5.png)

## 📊 Dataset

**Credit Card Fraud Detection** — publicly available on [Kaggle (Machine Learning Group, ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Fraud share | ~0.173% |
| Features | `Time`, `Amount`, `V1`–`V28` |
| Target | `Class` (`0` = legitimate, `1` = fraud) |

`V1`–`V28` are anonymized principal-component features. The dataset is heavily imbalanced, making precision, recall, F1-score and precision–recall AUC more informative than accuracy alone.

**Dataset attribution:** The data originates from the Machine Learning Group at Université Libre de Bruxelles (ULB) and is distributed through Kaggle. The original `creditcard.csv` is not included in this repository; download it from the linked dataset page.

## ✨ Features

- Batch transaction predictions from uploaded CSV files
- XGBoost-based fraud risk scoring
- Interactive dashboard with fraud alerts and charts
- Transaction explorer with risk categories and filtering
- SHAP explanations showing influential anonymized features
- Exportable prediction results

## 🧠 Model & Evaluation

The model uses **XGBoost** with class weighting to address the rare fraud class. Data was split using an 80/20 stratified train/test split.

| Metric | Result |
|---|---:|
| Precision–Recall AUC | 0.8327 |
| Fraud precision at threshold 0.5 | 0.35 |
| Fraud recall at threshold 0.5 | 0.87 |
| Fraud F1-score at threshold 0.5 | 0.49 |
| Fraud precision at threshold 0.7 | 0.54 |
| Fraud recall at threshold 0.7 | 0.86 |
| Fraud F1-score at threshold 0.7 | 0.66 |

**Evaluation note:** The 0.7 threshold was explored using the test set, so those results should be considered experimental rather than an unbiased final assessment. A more rigorous evaluation would select the threshold using a validation set and report performance on a separate untouched test set. Model probabilities are not necessarily calibrated.

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Streamlit · Plotly

## 🚀 Run Locally

1. Clone or download this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure compatible trained `fraud_model.pkl` and `features.pkl` files are available in the project folder. If these are not included, train the model using the provided training code/notebook first.
4. Start the application:

   ```bash
   streamlit run app.py
   ```

5. Upload a CSV containing the model's 30 input columns (`Time`, `Amount`, `V1`–`V28`).

## 📁 Project Structure

```text
FraudShield-AI/
├── app.py
├── README.md
├── requirements.txt
├── screenshots/
│   ├── dashboard.png
│   └── analytics.png
├── .streamlit/
│   └── config.toml
└── [training notebook / model files, if provided]
```

## ⚠️ Limitations

- Intended for learning and demonstration, not production banking decisions.
- The dataset is anonymized and historical; performance may change on new transactions.
- SHAP explains model feature contributions, not the underlying real-world meaning of anonymized features.
- Fraud alerts require human review; false positives and missed fraud cases are possible.

## 📚 Data Source

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

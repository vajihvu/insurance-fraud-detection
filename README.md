# insurance-fraud-detection

This repository contains an end-to-end machine learning pipeline to detect fraudulent insurance claims using tabular datasets. It demonstrates preprocessing, imbalance handling with SMOTE, model training (XGBoost & LightGBM), evaluation, and model explainability using SHAP.

## Contents

- `notebooks/insurance_fraud_detection.ipynb` — Main notebook with the complete workflow.
- `src/` — (optional) Modular scripts for preprocessing, training and evaluation.
- `data/` — Place the original dataset here.
- `models/` — Saved artifacts (e.g., `preprocessor.joblib`, `xgb_model.joblib`, `lgb_model.joblib`).
- `requirements.txt` — Python dependencies.

# insurance-fraud-detection

This repository contains an end-to-end machine learning pipeline to detect fraudulent insurance claims using tabular datasets. It demonstrates preprocessing, imbalance handling with SMOTE, model training (XGBoost & LightGBM), evaluation, and model explainability using SHAP.

## Contents

- `notebooks/insurance_fraud_detection.ipynb` — Main notebook with the complete workflow.
- `src/` — (optional) Modular scripts for preprocessing, training and evaluation.
- `data/` — Place the original dataset here.
- `models/` — Saved artifacts (e.g., `preprocessor.joblib`, `xgb_model.joblib`, `lgb_model.joblib`).
- `requirements.txt` — Python dependencies.

## Key Features
- Robust preprocessing: imputation, scaling and one-hot encoding via ColumnTransformer.
- Imbalance handling using SMOTE to improve minority-class (fraud) detection.
- Training of XGBoost and LightGBM models with evaluation using:
    - ROC-AUC
    - Precision-Recall AUC (PR-AUC / Average Precision)
    - F1-score, Confusion Matrix and Classification Report
- Explainability using SHAP summary plots to surface influential features.
- Saveable artifacts for deployment (joblib).

## Evaluation & Monitoring
- Prefer PR-AUC / Average Precision for imbalanced fraud tasks.
- Tune decision thresholds to balance business costs: false positives (investigation cost) vs false negatives (missed fraud).
- Consider model calibration and post-training auditing (SHAP + manual review).

## Next steps / Improvements
- Use SMOTENC or advanced encodings for mixed categorical features.
- Add time-aware validation (if claims are time-series).
- Hyperparameter optimization with Optuna.
- Package as an API using FastAPI, containerize with Docker, and add CI/CD & model monitoring (MLflow).

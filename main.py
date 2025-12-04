import os
import argparse
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
from src.data_preprocessing import load_data, detect_label, preprocess_split
from src.model_training import train_xgb, train_lgb
from src.evaluate import evaluate as evaluate_model
from src.utils import save_joblib, set_seed, get_logger

def parse_args():
    p = argparse.ArgumentParser(description="Insurance Claim Fraud Detection - main runner")
    p.add_argument("--data_path", required=True, help="Path to CSV dataset")
    p.add_argument("--label_col", default=None, help="Name of label column (if not auto-detected)")
    p.add_argument("--out_dir", default="models", help="Directory to save models and artifacts")
    p.add_argument("--test_size", type=float, default=0.2, help="Test set proportion")
    p.add_argument("--random_state", type=int, default=42, help="Random seed")
    p.add_argument("--do_shap", action="store_true", help="Compute and save SHAP summary plot (may be slow)")
    p.add_argument("--shap_max_samples", type=int, default=500, help="Max samples to use for SHAP explainer")
    p.add_argument("--smote", action="store_true", help="Apply SMOTE to training data (recommended)")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for evaluation")
    return p.parse_args()

def main():
    args = parse_args()
    logger = get_logger("main")
    set_seed(args.random_state)

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info(f"Loading data from: {args.data_path}")
    df = load_data(args.data_path)

    label_col = args.label_col
    if label_col is None:
        try:
            label_col = detect_label(df)
            logger.info(f"Auto-detected label column: {label_col}")
        except ValueError as e:
            logger.error(str(e))
            raise

    logger.info("Preprocessing data and creating train/test split...")
    X_train_prep, X_test_prep, y_train, y_test, preprocessor = preprocess_split(
        df, label_col, test_size=args.test_size, random_state=args.random_state
    )

    logger.info(f"Train shape (after preprocess): {X_train_prep.shape}")
    logger.info(f"Test shape (after preprocess): {X_test_prep.shape}")
    logger.info(f"Train class distribution:\n{y_train.value_counts().to_dict()}")

    if args.smote:
        logger.info("Applying SMOTE to training data...")
        sm = SMOTE(random_state=args.random_state)
        X_train_res, y_train_res = sm.fit_resample(X_train_prep, y_train)
        logger.info(f"After SMOTE class distribution:\n{np.bincount(y_train_res)}")
    else:
        logger.info("Skipping SMOTE (training on original imbalanced data).")
        X_train_res, y_train_res = X_train_prep, y_train

    logger.info("Training XGBoost...")
    xgb_model = train_xgb(X_train_res, y_train_res)
    logger.info("Training LightGBM...")
    lgb_model = train_lgb(X_train_res, y_train_res)

    logger.info(f"Saving models and preprocessor to {args.out_dir} ...")
    save_joblib(preprocessor, os.path.join(args.out_dir, "preprocessor.joblib"))
    save_joblib(xgb_model, os.path.join(args.out_dir, "xgb_model.joblib"))
    save_joblib(lgb_model, os.path.join(args.out_dir, "lgb_model.joblib"))
    logger.info("Artifacts saved.")

    logger.info("Evaluating XGBoost on test set...")
    evaluate_model(xgb_model, X_test_prep, y_test, name="XGBoost")
    logger.info("Evaluating LightGBM on test set...")
    evaluate_model(lgb_model, X_test_prep, y_test, name="LightGBM")

    if args.do_shap:
        try:
            logger.info("Computing SHAP values for XGBoost (this may take a while)...")
            n_samples = min(args.shap_max_samples, X_test_prep.shape[0])
            X_shap = X_test_prep[:n_samples]
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_shap)
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, X_shap, show=False)
            shap_path = os.path.join(args.out_dir, "shap_summary_xgb.png")
            plt.tight_layout()
            plt.savefig(shap_path, dpi=150)
            plt.close()
            logger.info(f"Saved SHAP summary plot to {shap_path}")
        except Exception as e:
            logger.exception("Failed to compute/save SHAP plot: %s", e)

    logger.info("Run completed successfully.")

if __name__ == "__main__":
    main()
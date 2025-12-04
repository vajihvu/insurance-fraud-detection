from typing import Dict
import joblib
from xgboost import XGBClassifier
import lightgbm as lgb
import os


def train_xgb(X, y, params: Dict = None):
    if params is None:
        params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8}
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
    model.set_params(**params)
    model.fit(X, y)
    return model

def train_lgb(X, y, params: Dict = None):
    if params is None:
        params = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31}
    model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    model.set_params(**params)
    model.fit(X, y)
    return model

def fit_and_save_models(X_train, y_train, preprocessor, out_dir: str = "./models"):
    os.makedirs(out_dir, exist_ok=True)

    xgb_model = train_xgb(X_train, y_train)
    lgb_model = train_lgb(X_train, y_train)

    joblib.dump(preprocessor, os.path.join(out_dir, "preprocessor.joblib"))
    joblib.dump(xgb_model, os.path.join(out_dir, "xgb_model.joblib"))
    joblib.dump(lgb_model, os.path.join(out_dir, "lgb_model.joblib"))

    return xgb_model, lgb_model
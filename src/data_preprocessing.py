from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def detect_label(df: pd.DataFrame, candidates: List[str] = None) -> str:

    if candidates is None:
        candidates = ["fraud_reported", "fraud", "is_fraud", "fraudulent", "label"]

    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No label column found. Checked: {candidates}")

def _split_numeric_categorical(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols

def build_preprocessor(
    X: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> ColumnTransformer:
    num_cols, cat_cols = _split_numeric_categorical(X)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=numeric_strategy)),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=categorical_strategy)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_transformer, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

def preprocess_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    
    df = df.copy()
    drop_cols = [c for c in ["policy_number", "policy_id", "claim_id", "policy_no", "id"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)

    preprocessor = build_preprocessor(X_train)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    return X_train_prep, X_test_prep, y_train, y_test, preprocessor
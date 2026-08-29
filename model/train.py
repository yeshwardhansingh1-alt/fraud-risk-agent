"""
Day 6 — First Model: LightGBM classifier.

Chronological train/val/test split (no random shuffling).
Train a first LightGBM, check raw AUC / PR-AUC.
Save the trained model artifact.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib
import os
import json

FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")
MODEL_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


# Columns to exclude from features
EXCLUDE_COLS = ["TransactionID", "TransactionDT", "isFraud"]


def chronological_split(df, train_frac=0.6, val_frac=0.2):
    """
    Strict chronological split — never random.
    Data must already be sorted by TransactionDT.

    Returns train, val, test DataFrames.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(f"Chronological split:")
    print(f"  Train: {len(train):,} rows  (TransactionDT {train['TransactionDT'].min():.0f} – {train['TransactionDT'].max():.0f})")
    print(f"  Val:   {len(val):,} rows  (TransactionDT {val['TransactionDT'].min():.0f} – {val['TransactionDT'].max():.0f})")
    print(f"  Test:  {len(test):,} rows  (TransactionDT {test['TransactionDT'].min():.0f} – {test['TransactionDT'].max():.0f})")

    # Confirm no overlap
    assert train["TransactionDT"].max() <= val["TransactionDT"].min(), "Train/Val overlap!"
    assert val["TransactionDT"].max() <= test["TransactionDT"].min(), "Val/Test overlap!"

    # Fraud rates
    print(f"\n  Train fraud rate: {train['isFraud'].mean():.4%}")
    print(f"  Val fraud rate:   {val['isFraud'].mean():.4%}")
    print(f"  Test fraud rate:  {test['isFraud'].mean():.4%}")

    return train, val, test


def get_feature_cols(df):
    """Get feature column names (everything except excluded cols)."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def train_lightgbm(train, val):
    """Train a LightGBM classifier. Returns model and feature columns."""
    feature_cols = get_feature_cols(train)
    print(f"\nTraining LightGBM with {len(feature_cols)} features...")

    X_train = train[feature_cols]
    y_train = train["isFraud"]
    X_val = val[feature_cols]
    y_val = val["isFraud"]

    # LightGBM parameters — conservative, not tuned yet (Day 6 = just confirm pipeline works)
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "is_unbalance": True,  # Handle class imbalance
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50),
        ],
    )

    return model, feature_cols


def evaluate_model(model, df, feature_cols, split_name="test"):
    """Evaluate model on a split. Returns predictions and metrics."""
    X = df[feature_cols]
    y = df["isFraud"]

    y_pred_proba = model.predict(X)

    roc_auc = roc_auc_score(y, y_pred_proba)
    pr_auc = average_precision_score(y, y_pred_proba)

    print(f"\n{split_name} metrics:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")

    return y_pred_proba, {"roc_auc": roc_auc, "pr_auc": pr_auc}


if __name__ == "__main__":
    # Load modeling table
    print("Loading modeling table...")
    df = pd.read_csv(os.path.join(FEATURES_DIR, "modeling_table.csv"))
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Chronological split
    train, val, test = chronological_split(df)

    # Train model
    model, feature_cols = train_lightgbm(train, val)

    # Evaluate on val and test
    val_preds, val_metrics = evaluate_model(model, val, feature_cols, "Validation")
    test_preds, test_metrics = evaluate_model(model, test, feature_cols, "Test")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lgbm_model.txt")
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save feature columns for later use
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    # Save metrics
    metrics = {"val": val_metrics, "test": test_metrics}
    with open(os.path.join(MODEL_DIR, "raw_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: model/raw_metrics.json")

    # Save split indices for reproducibility
    split_info = {
        "train_end_dt": float(train["TransactionDT"].max()),
        "val_end_dt": float(val["TransactionDT"].max()),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
    }
    with open(os.path.join(MODEL_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print("\nDay 6 complete.")

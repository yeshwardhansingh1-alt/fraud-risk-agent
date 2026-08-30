"""
Day 6 — First Model: LightGBM classifier.

Chronological train/val/test split (no random shuffling).
Train a first LightGBM, check raw AUC / PR-AUC.
Save the trained model artifact.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib
import os
import json
import optuna
import matplotlib.pyplot as plt
import datetime

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

    logger.info(f"Chronological split:")
    logger.info(f"  Train: {len(train):,} rows  (TransactionDT {train['TransactionDT'].min():.0f} – {train['TransactionDT'].max():.0f})")
    logger.info(f"  Val:   {len(val):,} rows  (TransactionDT {val['TransactionDT'].min():.0f} – {val['TransactionDT'].max():.0f})")
    logger.info(f"  Test:  {len(test):,} rows  (TransactionDT {test['TransactionDT'].min():.0f} – {test['TransactionDT'].max():.0f})")

    # Confirm no overlap
    assert train["TransactionDT"].max() <= val["TransactionDT"].min(), "Train/Val overlap!"
    assert val["TransactionDT"].max() <= test["TransactionDT"].min(), "Val/Test overlap!"

    # Fraud rates
    logger.info(f"\n  Train fraud rate: {train['isFraud'].mean():.4%}")
    logger.info(f"  Val fraud rate:   {val['isFraud'].mean():.4%}")
    logger.info(f"  Test fraud rate:  {test['isFraud'].mean():.4%}")

    return train, val, test

def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=10):
    """Run Optuna to find best hyperparameters."""
    logger.info(f"\nRunning Optuna hyperparameter tuning ({n_trials} trials)...")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1,
            "is_unbalance": True,
            "feature_pre_filter": False,
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        # Optuna minimizes by default, so return negative AUC
        y_pred = model.predict(X_val)
        return -roc_auc_score(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    logger.info(f"Best tuning params: {best_params}")
    
    # Merge best params with base params
    final_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_jobs": -1,
        "is_unbalance": True,
    }
    final_params.update(best_params)
    return final_params


def get_feature_cols(df):
    """Get feature column names (everything except excluded cols)."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def train_lightgbm(train, val, tune=True):
    """Train a LightGBM classifier. Returns model, feature columns, and params."""
    feature_cols = get_feature_cols(train)
    logger.info(f"\nTraining LightGBM with {len(feature_cols)} features...")

    X_train = train[feature_cols]
    y_train = train["isFraud"]
    X_val = val[feature_cols]
    y_val = val["isFraud"]

    if tune:
        params = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=10)
    else:
        # Default conservative params
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "verbose": -1,
            "n_jobs": -1,
            "is_unbalance": True,
        }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50),
        ],
    )

    return model, feature_cols, params


def evaluate_model(model, df, feature_cols, split_name="test"):
    """Evaluate model on a split. Returns predictions and metrics."""
    X = df[feature_cols]
    y = df["isFraud"]

    y_pred_proba = model.predict(X)

    roc_auc = roc_auc_score(y, y_pred_proba)
    pr_auc = average_precision_score(y, y_pred_proba)

    logger.info(f"\n{split_name} metrics:")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  PR-AUC:  {pr_auc:.4f}")

    return y_pred_proba, {"roc_auc": roc_auc, "pr_auc": pr_auc}


if __name__ == "__main__":
    # Load modeling table
    logger.info("Loading modeling table...")
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "modeling_table.parquet"))
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Chronological split
    train, val, test = chronological_split(df)

    # Train model
    model, feature_cols, best_params = train_lightgbm(train, val, tune=True)

    # Evaluate on val and test
    val_preds, val_metrics = evaluate_model(model, val, feature_cols, "Validation")
    test_preds, test_metrics = evaluate_model(model, test, feature_cols, "Test")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lgbm_model.txt")
    model.save_model(model_path)
    logger.info(f"\nModel saved to: {model_path}")

    # Save feature columns for later use
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    # Save metrics
    metrics = {"val": val_metrics, "test": test_metrics}
    with open(os.path.join(MODEL_DIR, "raw_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: model/raw_metrics.json")

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

    # 23. Feature Importance Analysis
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Save CSV
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split")
    }).sort_values("importance_gain", ascending=False)
    importance_df.to_parquet(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    
    # Save Plot
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(model, max_num_features=20, importance_type="gain", title="Top 20 Features (Gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=300)
    plt.close()
    logger.info(f"Saved feature importance plot to {PLOTS_DIR}/feature_importance.png")

    # 24. Model Versioning / Metadata Tracking
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": "LightGBM",
        "hyperparameters": best_params,
        "data_stats": split_info,
        "metrics": metrics,
        "best_iteration": model.best_iteration
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata tracking to model/metadata.json")



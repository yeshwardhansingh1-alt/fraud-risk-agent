"""
Day 15 — Chronological Backtest [CORE].

Confirm train/val/test are strictly time-ordered with no leakage.
Compute precision, recall, ROC-AUC, PR-AUC on the test slice.
Recompute false-positive cost and missed-fraud cost.
This is literally "measured precision and recall on a held-out test set."
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import json
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")
from model.cost_model import compute_costs, print_cost_report, COST_CONFIG

MODEL_DIR = os.path.dirname(__file__)
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def verify_no_leakage(train, val, test):
    """Confirm train/val/test are strictly time-ordered with no overlap."""
    print("Verifying chronological ordering...")

    train_max = train["TransactionDT"].max()
    val_min = val["TransactionDT"].min()
    val_max = val["TransactionDT"].max()
    test_min = test["TransactionDT"].min()

    assert train_max <= val_min, f"LEAKAGE: train max ({train_max}) > val min ({val_min})"
    assert val_max <= test_min, f"LEAKAGE: val max ({val_max}) > test min ({test_min})"

    print(f"  [OK] Train: TransactionDT [{train['TransactionDT'].min():.0f}, {train_max:.0f}]")
    print(f"  [OK] Val:   TransactionDT [{val_min:.0f}, {val_max:.0f}]")
    print(f"  [OK] Test:  TransactionDT [{test_min:.0f}, {test['TransactionDT'].max():.0f}]")
    print(f"  [OK] No time leakage detected.")


def full_evaluation(y_true, y_pred_proba, threshold=0.5):
    """Compute all evaluation metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "n_total": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "n_predicted_fraud": int(y_pred.sum()),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["true_negatives"] = int(cm[0, 0])
    metrics["false_positives"] = int(cm[0, 1])
    metrics["false_negatives"] = int(cm[1, 0])
    metrics["true_positives"] = int(cm[1, 1])

    return metrics


def plot_evaluation_charts(y_true, y_pred_proba, save_dir):
    """Plot ROC curve, PR curve, and score distribution."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    axes[0].plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    axes[1].plot(recall, precision, color="#4CAF50", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    axes[1].axhline(y_true.mean(), color="gray", linestyle="--", label="Baseline (fraud rate)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (more informative than ROC for rare fraud)")
    axes[1].legend()

    # 3. Score Distribution
    fraud_scores = y_pred_proba[y_true == 1]
    legit_scores = y_pred_proba[y_true == 0]
    axes[2].hist(legit_scores, bins=50, alpha=0.5, color="#2196F3", label="Legit", density=True)
    axes[2].hist(fraud_scores, bins=50, alpha=0.5, color="#F44336", label="Fraud", density=True)
    axes[2].axvline(0.5, color="black", linestyle="--", label="Threshold=0.5")
    axes[2].set_xlabel("Predicted Fraud Probability")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Score Distribution")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "evaluation_charts.png"), dpi=150)
    plt.close()
    print(f"  Saved evaluation charts to: {save_dir}/evaluation_charts.png")


if __name__ == "__main__":
    # Load data
    print("Loading modeling table...")
    df = pd.read_csv(os.path.join(FEATURES_DIR, "modeling_table.csv"))
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split_info = json.load(f)

    # Recreate splits
    train_end = split_info["train_size"]
    val_end = train_end + split_info["val_size"]
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    # Verify no leakage
    verify_no_leakage(train, val, test)

    # Load calibrated model
    calibrated_model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))

    # Predict on held-out test set
    print("\nPredicting on held-out test set...")
    y_true = test["isFraud"].values
    y_pred_proba = calibrated_model.predict_proba(test[feature_cols])[:, 1]

    # Full evaluation metrics
    metrics = full_evaluation(y_true, y_pred_proba)

    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET RESULTS")
    print("=" * 60)
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"\n  True Positives:  {metrics['true_positives']:,}")
    print(f"  False Positives: {metrics['false_positives']:,}")
    print(f"  False Negatives: {metrics['false_negatives']:,}")
    print(f"  True Negatives:  {metrics['true_negatives']:,}")

    # Cost-aware metrics (Day 8 recomputed on test set)
    print("\n--- Cost-Aware Metrics on Test Set ---")
    cost_results = compute_costs(test, y_true, y_pred_proba)
    print_cost_report(cost_results)

    # Save metrics
    all_results = {
        "classification_metrics": metrics,
        "cost_metrics": cost_results,
    }
    with open(os.path.join(MODEL_DIR, "backtest_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved backtest results to: model/backtest_results.json")

    # Plot evaluation charts
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_evaluation_charts(y_true, y_pred_proba, PLOTS_DIR)

    print("\nDay 15 complete — this is 'measured precision and recall on a held-out test set.'")

"""
Day 17 — Error Analysis [STRETCH].

Re-check reliability diagram on the test slice specifically.
Look at false positives and false negatives manually — note patterns.
Write down 2-3 concrete failure cases for the write-up.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import json
import joblib
import os
import sys


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def analyze_errors(test, y_true, y_pred_proba, feature_cols, threshold=0.5):
    """Analyze false positives and false negatives."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    fps = test[fp_mask].copy()
    fns = test[fn_mask].copy()
    fps["predicted_proba"] = y_pred_proba[fp_mask]
    fns["predicted_proba"] = y_pred_proba[fn_mask]

    logger.info("=" * 60)
    logger.info("ERROR ANALYSIS")
    logger.info("=" * 60)

    # --- False Positives ---
    logger.info(f"\n--- FALSE POSITIVES ({len(fps):,}) ---")
    logger.info(f"  These are legitimate transactions we incorrectly blocked.")
    logger.info(f"\n  Amount stats:")
    logger.info(f"    Mean: ${fps['TransactionAmt'].mean():.2f}")
    logger.info(f"    Median: ${fps['TransactionAmt'].median():.2f}")
    logger.info(f"    Max: ${fps['TransactionAmt'].max():.2f}")
    logger.info(f"  Predicted probability stats:")
    logger.info(f"    Mean: {fps['predicted_proba'].mean():.4f}")
    logger.info(f"    Median: {fps['predicted_proba'].median():.4f}")

    # Common patterns in FPs
    if "hour_of_day" in fps.columns:
        top_hours = fps["hour_of_day"].value_counts().head(3)
        logger.info(f"\n  Most common hours for FPs: {dict(top_hours)}")

    if "card1_txn_count_24hr" in fps.columns:
        logger.info(f"  Average velocity (txn/24hr): {fps['card1_txn_count_24hr'].mean():.1f}")

    # --- False Negatives ---
    logger.info(f"\n--- FALSE NEGATIVES ({len(fns):,}) ---")
    logger.info(f"  These are fraud we missed (approved fraudulent transactions).")
    logger.info(f"\n  Amount stats:")
    logger.info(f"    Mean: ${fns['TransactionAmt'].mean():.2f}")
    logger.info(f"    Median: ${fns['TransactionAmt'].median():.2f}")
    logger.info(f"    Max: ${fns['TransactionAmt'].max():.2f}")
    logger.info(f"  Predicted probability stats:")
    logger.info(f"    Mean: {fns['predicted_proba'].mean():.4f}")
    logger.info(f"    Max: {fns['predicted_proba'].max():.4f}")

    # --- Concrete Failure Cases ---
    logger.info(f"\n{'='*60}")
    logger.info("CONCRETE FAILURE CASES")
    logger.info("=" * 60)

    # FP failure case: highest-confidence false positive
    if len(fps) > 0:
        worst_fp = fps.nlargest(1, "predicted_proba").iloc[0]
        logger.info(f"\n  Failure Case 1: Highest-Confidence False Positive")
        logger.info(f"    TransactionID: {worst_fp['TransactionID']:.0f}")
        logger.info(f"    Amount: ${worst_fp['TransactionAmt']:.2f}")
        logger.info(f"    Predicted fraud prob: {worst_fp['predicted_proba']:.4f}")
        logger.info(f"    Was actually: LEGITIMATE")
        logger.info(f"    Impact: customer blocked, ${worst_fp['TransactionAmt']:.2f} revenue at risk")

    # FP failure case: highest-amount false positive
    if len(fps) > 0:
        expensive_fp = fps.nlargest(1, "TransactionAmt").iloc[0]
        logger.info(f"\n  Failure Case 2: Highest-Amount False Positive")
        logger.info(f"    TransactionID: {expensive_fp['TransactionID']:.0f}")
        logger.info(f"    Amount: ${expensive_fp['TransactionAmt']:.2f}")
        logger.info(f"    Predicted fraud prob: {expensive_fp['predicted_proba']:.4f}")
        logger.info(f"    Was actually: LEGITIMATE")
        logger.info(f"    Impact: ${expensive_fp['TransactionAmt'] * 0.80:.2f} estimated lost revenue")

    # FN failure case: highest-amount missed fraud
    if len(fns) > 0:
        worst_fn = fns.nlargest(1, "TransactionAmt").iloc[0]
        logger.info(f"\n  Failure Case 3: Highest-Amount Missed Fraud")
        logger.info(f"    TransactionID: {worst_fn['TransactionID']:.0f}")
        logger.info(f"    Amount: ${worst_fn['TransactionAmt']:.2f}")
        logger.info(f"    Predicted fraud prob: {worst_fn['predicted_proba']:.4f}")
        logger.info(f"    Was actually: FRAUD")
        logger.info(f"    Impact: ${worst_fn['TransactionAmt'] + 25:.2f} total loss (amount + chargeback)")

    return fps, fns


def plot_test_reliability_diagram(y_true, y_pred_proba, save_path):
    """Reliability diagram specifically on the test slice."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=20, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", color="#2196F3",
            label="Calibrated Model (Test Set)")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Fraud Rate")
    ax.set_title("Reliability Diagram — Test Set Only")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"  Saved: {save_path}")


if __name__ == "__main__":
    # Load data
    logger.info("Loading data...")
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "modeling_table.parquet"))
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split_info = json.load(f)

    test = df.iloc[split_info["train_size"] + split_info["val_size"]:]
    y_true = test["isFraud"].values

    calibrated_model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))
    y_pred_proba = calibrated_model.predict_proba(test[feature_cols])[:, 1]

    # Reliability diagram on test set
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_test_reliability_diagram(
        y_true, y_pred_proba,
        os.path.join(PLOTS_DIR, "reliability_diagram_test.png"),
    )

    # Error analysis
    fps, fns = analyze_errors(test, y_true, y_pred_proba, feature_cols)



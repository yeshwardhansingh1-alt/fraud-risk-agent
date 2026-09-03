"""
Day 7 — Calibration.

Wrap the LightGBM model in CalibratedClassifierCV (isotonic),
plot reliability diagram, compute Brier score.
This is the P(Loss|X) engine.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import joblib
import json
import os

FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")
MODEL_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


import sys

from model.wrappers import LGBMWrapper, CustomCalibratedClassifier




def calibrate_model(model_wrapper, X_val, y_val):
    """
    Calibrate the model using isotonic regression on the validation set.
    """
    logger.info("Calibrating model (isotonic regression)...")
    calibrated = CustomCalibratedClassifier(model_wrapper)
    calibrated.fit(X_val, y_val)
    return calibrated


def plot_reliability_diagram(y_true, y_pred_uncal, y_pred_cal, save_path):
    """Plot reliability diagram: predicted probability vs. observed fraud rate."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, y_pred, title in [
        (axes[0], y_pred_uncal, "Before Calibration (Raw LightGBM)"),
        (axes[1], y_pred_cal, "After Calibration (Isotonic)"),
    ]:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred, n_bins=20, strategy="quantile"
        )
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", color="#2196F3", label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Fraud Rate")
        ax.set_title(title)
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"  Saved reliability diagram: {save_path}")


if __name__ == "__main__":
    # Load data
    logger.info("Loading modeling table...")
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "modeling_table.parquet"))
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Load model and feature cols
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm_model.txt"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    # Load split info
    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split_info = json.load(f)

    # Recreate splits
    train_end = split_info["train_size"]
    val_end = train_end + split_info["val_size"]

    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    X_val = val[feature_cols]
    y_val = val["isFraud"]
    X_test = test[feature_cols]
    y_test = test["isFraud"]

    # Uncalibrated predictions
    y_pred_uncal_val = booster.predict(X_val)
    y_pred_uncal_test = booster.predict(X_test)

    # Brier score before calibration
    brier_before = brier_score_loss(y_test, y_pred_uncal_test)
    logger.info(f"\nBrier score BEFORE calibration: {brier_before:.6f}")

    # Calibrate
    model_wrapper = LGBMWrapper(booster, feature_cols)
    calibrated_model = calibrate_model(model_wrapper, X_val, y_val)

    # Calibrated predictions
    y_pred_cal_val = calibrated_model.predict_proba(X_val)[:, 1]
    y_pred_cal_test = calibrated_model.predict_proba(X_test)[:, 1]

    # Brier score after calibration
    brier_after = brier_score_loss(y_test, y_pred_cal_test)
    logger.info(f"Brier score AFTER calibration:  {brier_after:.6f}")
    logger.info(f"Improvement: {(brier_before - brier_after) / brier_before:.2%}")

    # Reliability diagram
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_reliability_diagram(
        y_test, y_pred_uncal_test, y_pred_cal_test,
        os.path.join(PLOTS_DIR, "reliability_diagram.png"),
    )

    # Save calibrated model
    cal_model_path = os.path.join(MODEL_DIR, "calibrated_model.pkl")
    joblib.dump(calibrated_model, cal_model_path)
    logger.info(f"\nCalibrated model saved to: {cal_model_path}")

    # Save calibration metrics
    cal_metrics = {
        "brier_before": float(brier_before),
        "brier_after": float(brier_after),
        "improvement_pct": float((brier_before - brier_after) / brier_before * 100),
    }
    with open(os.path.join(MODEL_DIR, "calibration_metrics.json"), "w") as f:
        json.dump(cal_metrics, f, indent=2)



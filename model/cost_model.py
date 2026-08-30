"""
Day 8 — Cost Model [CORE].

Define cost assumptions: chargeback fee, false-positive friction cost,
missed-fraud cost. Compute false-positive and missed-fraud cost on validation set.

"Honest metrics including false-positive cost" — this is the exact line from the bar.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import json
import os

import sys


MODEL_DIR = os.path.dirname(__file__)
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")

# ============================================================
# COST ASSUMPTIONS
# ============================================================
# Document every assumption inline — "assumption, tune with real data if available"
# is an honest and completely fine thing to write.

CONFIG_PATH = os.path.join(MODEL_DIR, "cost_config.json")

def load_cost_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {
        "false_positive_friction_cost": 25.0,
        "lost_revenue_fraction": 0.80,
        "chargeback_fee": 25.0,
        "fraud_loss_fraction": 1.0,
        "step_up_leakage_rate": 0.25,
        "step_up_friction_cost": 5.0,
        "dispute_processing_fee": 10.0,
        "threshold": 0.5,
    }

COST_CONFIG = load_cost_config()


def compute_costs(df, y_true, y_pred_proba, config=None):
    """
    Compute false-positive and missed-fraud costs on a dataset.

    Args:
        df: DataFrame with TransactionAmt column
        y_true: actual fraud labels (0/1)
        y_pred_proba: predicted fraud probabilities
        config: cost config dict (uses COST_CONFIG if None)

    Returns:
        dict with cost breakdown
    """
    if config is None:
        config = COST_CONFIG

    threshold = config["threshold"]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix components
    tp = ((y_pred == 1) & (y_true == 1))  # True positives (correctly flagged fraud)
    fp = ((y_pred == 1) & (y_true == 0))  # False positives (blocked legit)
    fn = ((y_pred == 0) & (y_true == 1))  # False negatives (missed fraud)
    tn = ((y_pred == 0) & (y_true == 0))  # True negatives (correctly approved)

    amounts = df["TransactionAmt"].values

    # --- False Positive Costs ---
    fp_friction = fp.sum() * config["false_positive_friction_cost"]
    fp_lost_revenue = (amounts[fp] * config["lost_revenue_fraction"]).sum()
    total_fp_cost = fp_friction + fp_lost_revenue

    # --- Missed Fraud Costs ---
    fn_chargeback = fn.sum() * config["chargeback_fee"]
    fn_fraud_loss = (amounts[fn] * config["fraud_loss_fraction"]).sum()
    total_fn_cost = fn_chargeback + fn_fraud_loss

    # --- Prevented Fraud Value ---
    prevented_fraud_value = (amounts[tp] * config["fraud_loss_fraction"]).sum()
    prevented_chargeback_fees = tp.sum() * config["chargeback_fee"]

    # --- Net ---
    total_cost = total_fp_cost + total_fn_cost
    net_savings = (prevented_fraud_value + prevented_chargeback_fees) - total_fp_cost

    results = {
        "threshold": threshold,
        "n_transactions": len(y_true),
        "n_fraud": int(y_true.sum()),
        "n_predicted_fraud": int(y_pred.sum()),

        # Counts
        "true_positives": int(tp.sum()),
        "false_positives": int(fp.sum()),
        "false_negatives": int(fn.sum()),
        "true_negatives": int(tn.sum()),

        # Rates
        "precision": float(tp.sum() / y_pred.sum()) if y_pred.sum() > 0 else 0,
        "recall": float(tp.sum() / y_true.sum()) if y_true.sum() > 0 else 0,

        # Costs
        "fp_friction_cost": float(fp_friction),
        "fp_lost_revenue": float(fp_lost_revenue),
        "total_false_positive_cost": float(total_fp_cost),

        "fn_chargeback_fees": float(fn_chargeback),
        "fn_fraud_loss": float(fn_fraud_loss),
        "total_missed_fraud_cost": float(total_fn_cost),

        # Savings
        "prevented_fraud_value": float(prevented_fraud_value),
        "prevented_chargeback_fees": float(prevented_chargeback_fees),
        "net_savings": float(net_savings),
        "total_cost": float(total_cost),

        # Per-transaction averages
        "false_positive_cost_per_1000_txn": float(total_fp_cost / len(y_true) * 1000),
        "missed_fraud_cost_per_1000_txn": float(total_fn_cost / len(y_true) * 1000),
    }

    return results


def find_optimal_threshold(df, y_true, y_pred_proba, config=None):
    """Sweep thresholds from 0.05 to 0.95 to find the one that minimizes total cost."""
    if config is None:
        config = COST_CONFIG.copy()
        
    best_threshold = 0.5
    min_cost = float("inf")
    
    logger.info("\nSweeping thresholds 0.05 -> 0.95...")
    for t in np.arange(0.05, 0.96, 0.01):
        config["threshold"] = float(t)
        res = compute_costs(df, y_true, y_pred_proba, config)
        if res["total_cost"] < min_cost:
            min_cost = res["total_cost"]
            best_threshold = float(t)
            
    return best_threshold


def print_cost_report(results):
    """Pretty-print the cost report."""
    logger.info("\n" + "=" * 60)
    logger.info("COST-AWARE METRICS REPORT")
    logger.info("=" * 60)

    logger.info(f"\nThreshold: {results['threshold']}")
    logger.info(f"Transactions: {results['n_transactions']:,}")
    logger.info(f"Actual fraud: {results['n_fraud']:,}")
    logger.info(f"Predicted fraud: {results['n_predicted_fraud']:,}")

    logger.info(f"\nPrecision: {results['precision']:.4f}")
    logger.info(f"Recall:    {results['recall']:.4f}")

    logger.info(f"\n--- False Positive Costs (blocking legitimate transactions) ---")
    logger.info(f"  False positives:    {results['false_positives']:,}")
    logger.info(f"  Friction cost:      ${results['fp_friction_cost']:,.2f}")
    logger.info(f"  Lost revenue:       ${results['fp_lost_revenue']:,.2f}")
    logger.info(f"  TOTAL FP cost:      ${results['total_false_positive_cost']:,.2f}")

    logger.info(f"\n--- Missed Fraud Costs (approving fraudulent transactions) ---")
    logger.info(f"  False negatives:    {results['false_negatives']:,}")
    logger.info(f"  Chargeback fees:    ${results['fn_chargeback_fees']:,.2f}")
    logger.info(f"  Fraud loss:         ${results['fn_fraud_loss']:,.2f}")
    logger.info(f"  TOTAL missed cost:  ${results['total_missed_fraud_cost']:,.2f}")

    logger.info(f"\n--- Summary ---")
    logger.info(f"  Total cost:                ${results['total_cost']:,.2f}")
    logger.info(f"  Prevented fraud value:     ${results['prevented_fraud_value']:,.2f}")
    logger.info(f"  Net savings:               ${results['net_savings']:,.2f}")
    logger.info(f"  FP cost per 1,000 txn:     ${results['false_positive_cost_per_1000_txn']:,.2f}")
    logger.info(f"  Missed fraud per 1,000 txn: ${results['missed_fraud_cost_per_1000_txn']:,.2f}")

    logger.info(f"\n--- Cost Assumptions (document inline) ---")
    logger.info(f"  Friction cost per FP:       ${COST_CONFIG['false_positive_friction_cost']}")
    logger.info(f"  Lost revenue fraction:      {COST_CONFIG['lost_revenue_fraction']:.0%}")
    logger.info(f"  Chargeback fee:             ${COST_CONFIG['chargeback_fee']}")
    logger.info(f"  Fraud loss fraction:         {COST_CONFIG['fraud_loss_fraction']:.0%}")
    logger.info(f"  Assumption: tune with real data if available.")

    logger.info("=" * 60)


if __name__ == "__main__":
    import joblib

    # Load data
    logger.info("Loading modeling table...")
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "modeling_table.parquet"))
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Load split info
    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split_info = json.load(f)
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    train_end = split_info["train_size"]
    val_end = train_end + split_info["val_size"]
    val = df.iloc[train_end:val_end]

    # Load calibrated model
    calibrated_model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))

    # Get calibrated predictions on validation set
    y_pred_proba = calibrated_model.predict_proba(val[feature_cols])[:, 1]
    y_true = val["isFraud"].values

    optimal_threshold = find_optimal_threshold(val, y_true, y_pred_proba)
    logger.info(f"Optimal cost-minimizing threshold: {optimal_threshold:.2f}")
    COST_CONFIG["threshold"] = optimal_threshold

    # Compute costs
    results = compute_costs(val, y_true, y_pred_proba)
    print_cost_report(results)

    # Save cost config and results
    with open(os.path.join(MODEL_DIR, "cost_config.json"), "w") as f:
        json.dump(COST_CONFIG, f, indent=2)

    with open(os.path.join(MODEL_DIR, "cost_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved cost config and results to model/")


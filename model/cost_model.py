"""
Day 8 — Cost Model [CORE].

Define cost assumptions: chargeback fee, false-positive friction cost,
missed-fraud cost. Compute false-positive and missed-fraud cost on validation set.

"Honest metrics including false-positive cost" — this is the exact line from the bar.
"""

import pandas as pd
import numpy as np
import json
import os

import sys
sys.path.insert(0, os.path.dirname(__file__) + "/..")

MODEL_DIR = os.path.dirname(__file__)
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")

# ============================================================
# COST ASSUMPTIONS
# ============================================================
# Document every assumption inline — "assumption, tune with real data if available"
# is an honest and completely fine thing to write.

COST_CONFIG = {
    # --- False Positive Costs (blocking a legitimate transaction) ---

    # Friction cost: customer frustration, checkout abandonment.
    # Assumption: average lost revenue per false positive = $25.
    # Real-world: varies by merchant (e-commerce ~2-5% of basket, travel ~$50+).
    # Tune with real data if available.
    "false_positive_friction_cost": 25.0,

    # Lost legitimate revenue: the transaction amount itself is lost.
    # We'll use a fraction of the transaction amount.
    # Assumption: 80% of the transaction value is lost (some customers retry).
    "lost_revenue_fraction": 0.80,

    # --- Missed Fraud Costs (approving a fraudulent transaction) ---

    # Chargeback fee: flat fee charged by the card network.
    # Assumption: $25 per chargeback (industry standard: $15-$100).
    "chargeback_fee": 25.0,

    # Penalty/margin loss: the full transaction amount is lost.
    # Assumption: 100% of the transaction amount.
    "fraud_loss_fraction": 1.0,

    # --- Decision threshold ---
    "threshold": 0.5,
}


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


def print_cost_report(results):
    """Pretty-print the cost report."""
    print("\n" + "=" * 60)
    print("COST-AWARE METRICS REPORT")
    print("=" * 60)

    print(f"\nThreshold: {results['threshold']}")
    print(f"Transactions: {results['n_transactions']:,}")
    print(f"Actual fraud: {results['n_fraud']:,}")
    print(f"Predicted fraud: {results['n_predicted_fraud']:,}")

    print(f"\nPrecision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")

    print(f"\n--- False Positive Costs (blocking legitimate transactions) ---")
    print(f"  False positives:    {results['false_positives']:,}")
    print(f"  Friction cost:      ${results['fp_friction_cost']:,.2f}")
    print(f"  Lost revenue:       ${results['fp_lost_revenue']:,.2f}")
    print(f"  TOTAL FP cost:      ${results['total_false_positive_cost']:,.2f}")

    print(f"\n--- Missed Fraud Costs (approving fraudulent transactions) ---")
    print(f"  False negatives:    {results['false_negatives']:,}")
    print(f"  Chargeback fees:    ${results['fn_chargeback_fees']:,.2f}")
    print(f"  Fraud loss:         ${results['fn_fraud_loss']:,.2f}")
    print(f"  TOTAL missed cost:  ${results['total_missed_fraud_cost']:,.2f}")

    print(f"\n--- Summary ---")
    print(f"  Total cost:                ${results['total_cost']:,.2f}")
    print(f"  Prevented fraud value:     ${results['prevented_fraud_value']:,.2f}")
    print(f"  Net savings:               ${results['net_savings']:,.2f}")
    print(f"  FP cost per 1,000 txn:     ${results['false_positive_cost_per_1000_txn']:,.2f}")
    print(f"  Missed fraud per 1,000 txn: ${results['missed_fraud_cost_per_1000_txn']:,.2f}")

    print(f"\n--- Cost Assumptions (document inline) ---")
    print(f"  Friction cost per FP:       ${COST_CONFIG['false_positive_friction_cost']}")
    print(f"  Lost revenue fraction:      {COST_CONFIG['lost_revenue_fraction']:.0%}")
    print(f"  Chargeback fee:             ${COST_CONFIG['chargeback_fee']}")
    print(f"  Fraud loss fraction:         {COST_CONFIG['fraud_loss_fraction']:.0%}")
    print(f"  Assumption: tune with real data if available.")

    print("=" * 60)


if __name__ == "__main__":
    import joblib

    # Load data
    print("Loading modeling table...")
    df = pd.read_csv(os.path.join(FEATURES_DIR, "modeling_table.csv"))
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

    # Compute costs
    results = compute_costs(val, y_true, y_pred_proba)
    print_cost_report(results)

    # Save cost config and results
    with open(os.path.join(MODEL_DIR, "cost_config.json"), "w") as f:
        json.dump(COST_CONFIG, f, indent=2)

    with open(os.path.join(MODEL_DIR, "cost_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved cost config and results to model/")
    print("Day 8 complete.")

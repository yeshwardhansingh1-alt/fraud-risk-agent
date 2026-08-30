"""
Day 10 — Decision Agent Wrapper [STRETCH].

Transaction → calibrated probability → argmin action → structured JSON output.
JSON includes: action, probability, expected loss per action, top contributing features.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import json
import joblib
import os
import sys

from agent.expected_loss import argmin_action, expected_loss, ACTIONS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")


def make_decision(calibrated_model, feature_cols, transaction_row, shap_values=None):
    """
    Process a single transaction through the full decision pipeline.

    Args:
        calibrated_model: sklearn CalibratedClassifierCV wrapping LightGBM
        feature_cols: list of feature column names
        transaction_row: dict or pd.Series with feature values
        shap_values: optional dict of {feature_name: shap_value}

    Returns:
        Structured decision dict (JSON-serializable)
    """
    # Extract features
    if isinstance(transaction_row, pd.Series):
        features = transaction_row[feature_cols].values.reshape(1, -1)
        txn_id = transaction_row.get("TransactionID", "unknown")
        txn_amt = transaction_row.get("TransactionAmt", 0)
    else:
        features = np.array([[transaction_row.get(c, 0) for c in feature_cols]])
        txn_id = transaction_row.get("TransactionID", "unknown")
        txn_amt = transaction_row.get("TransactionAmt", 0)

    # Get calibrated probability
    p_fraud = float(calibrated_model.predict_proba(features)[0, 1])

    # Find optimal action via expected loss minimization
    best_action, losses = argmin_action(p_fraud, txn_amt)

    # Build decision receipt
    decision = {
        "transaction_id": int(txn_id) if not isinstance(txn_id, str) else txn_id,
        "transaction_amount": float(txn_amt),
        "fraud_probability": round(p_fraud, 6),
        "action": best_action,
        "expected_losses": {a: round(v, 2) for a, v in losses.items()},
        "reason": _generate_reason(best_action, p_fraud, txn_amt, losses),
    }

    # Add top contributing features if SHAP values are provided
    if shap_values is not None:
        sorted_features = sorted(
            shap_values.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]
        decision["top_features"] = [
            {"feature": name, "shap_value": round(val, 4)}
            for name, val in sorted_features
        ]

    return decision


def _generate_reason(action, p_fraud, amount, losses):
    """Generate a human-readable decision reason (the 'decision receipt')."""
    best_loss = losses[action]

    # Compare against the next-best action
    other_losses = {a: v for a, v in losses.items() if a != action}
    next_best = min(other_losses, key=other_losses.get)
    next_loss = other_losses[next_best]
    savings = next_loss - best_loss

    reason = (
        f"{action.replace('_', '-').title()} chosen: "
        f"E[loss|{action}]=${best_loss:.2f} vs E[loss|{next_best}]=${next_loss:.2f}"
    )

    if savings > 0:
        reason += f" (saves ${savings:.2f})"

    return reason


def run_agent_on_dataset(calibrated_model, feature_cols, df, shap_explainer=None):
    """
    Run the decision agent over an entire dataset.

    Returns list of decision dicts.
    """
    logger.info(f"Running decision agent on {len(df):,} transactions...")
    decisions = []

    for idx, row in df.iterrows():
        # Optionally compute SHAP values per transaction
        shap_values = None
        if shap_explainer is not None:
            sv = shap_explainer(row[feature_cols].values.reshape(1, -1))
            shap_values = dict(zip(feature_cols, sv.values[0]))

        decision = make_decision(calibrated_model, feature_cols, row, shap_values)
        decisions.append(decision)

        if len(decisions) % 10000 == 0:
            logger.info(f"  Processed {len(decisions):,} transactions...")

    logger.info(f"  Done. {len(decisions):,} decisions generated.")
    return decisions


def summarize_decisions(decisions):
    """Print summary of agent decisions."""
    actions = [d["action"] for d in decisions]
    action_counts = pd.Series(actions).value_counts()

    logger.info("\n" + "=" * 60)
    logger.info("DECISION AGENT SUMMARY")
    logger.info("=" * 60)
    for action, count in action_counts.items():
        pct = count / len(decisions) * 100
        logger.info(f"  {action:15s}: {count:>8,} ({pct:.1f}%)")

    avg_prob = np.mean([d["fraud_probability"] for d in decisions])
    logger.info(f"\n  Average fraud probability: {avg_prob:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Load calibrated model
    logger.info("Loading calibrated model...")
    calibrated_model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split_info = json.load(f)

    # Load test set
    logger.info("Loading test data...")
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "modeling_table.parquet"))
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    test = df.iloc[split_info["train_size"] + split_info["val_size"]:]

    # Run agent (without SHAP for speed — SHAP is added in Day 12-13)
    decisions = run_agent_on_dataset(calibrated_model, feature_cols, test)
    summarize_decisions(decisions)

    # Save all decisions
    out_path = os.path.join(os.path.dirname(__file__), "test_decisions.json")
    with open(out_path, "w") as f:
        json.dump(decisions, f, indent=2)
    logger.info(f"\nSaved {len(decisions):,} decisions to: {out_path}")


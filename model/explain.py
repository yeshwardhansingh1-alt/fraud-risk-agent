"""
Days 12-13 — Explainability Layer [CORE].

LightGBM built-in SHAP functionality for feature-level attribution on flagged transactions.
Extract top 3-5 contributing features per flagged transaction.
Build "decision receipt" text generator that attaches to the agent's JSON output.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import json
import joblib
import sys
import os


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def explain_transaction(booster, features, feature_cols, top_n=5):
    """
    Explain a single transaction using LightGBM's native SHAP values.

    Args:
        booster: LightGBM Booster
        features: 1D array of feature values
        feature_cols: list of feature names
        top_n: number of top contributing features to return

    Returns:
        dict with SHAP values and top features
    """
    features_2d = np.array(features).reshape(1, -1)
    
    # LightGBM returns SHAP values + expected value (base value) as the last column
    shap_values_with_base = booster.predict(features_2d, pred_contrib=True)[0]
    sv = shap_values_with_base[:-1]
    base_val = shap_values_with_base[-1]

    # Sort by absolute SHAP value
    feature_importance = sorted(
        zip(feature_cols, sv, features.flatten()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    top_features = [
        {
            "feature": name,
            "shap_value": round(float(shap_val), 4),
            "feature_value": round(float(feat_val), 4) if not np.isnan(feat_val) else None,
            "direction": "increases fraud risk" if shap_val > 0 else "decreases fraud risk",
        }
        for name, shap_val, feat_val in feature_importance[:top_n]
    ]

    return {
        "shap_values": dict(zip(feature_cols, [round(float(v), 4) for v in sv])),
        "top_features": top_features,
        "base_value": round(float(base_val), 4),
    }


def generate_decision_receipt(transaction, decision, explanation):
    """
    Generate a structured "decision receipt" per transaction.
    This is the "honest, auditable" reason code.

    Format: "Step-Up chosen: E[loss|approve]=$46 vs E[loss|step-up]=$3"
    + SHAP feature explanations
    """
    receipt_lines = []

    # Decision summary
    action = decision.get("action", "unknown")
    prob = decision.get("fraud_probability", 0)
    receipt_lines.append(f"Decision: {action.upper()}")
    receipt_lines.append(f"Fraud probability: {prob:.4f}")
    receipt_lines.append(f"Reason: {decision.get('reason', 'N/A')}")

    # Expected losses
    losses = decision.get("expected_losses", {})
    if losses:
        receipt_lines.append("\nExpected losses by action:")
        for a, l in sorted(losses.items(), key=lambda x: x[1]):
            marker = " <- chosen" if a == action else ""
            receipt_lines.append(f"  {a:15s}: ${l:>10.2f}{marker}")

    # Top contributing features from SHAP
    if explanation and "top_features" in explanation:
        receipt_lines.append(f"\nTop {len(explanation['top_features'])} contributing features:")
        for feat in explanation["top_features"]:
            direction = "(+)" if feat["shap_value"] > 0 else "(-)"
            receipt_lines.append(
                f"  {direction} {feat['feature']:35s}  "
                f"value={feat['feature_value']}  "
                f"SHAP={feat['shap_value']:+.4f}  "
                f"({feat['direction']})"
            )

    return "\n".join(receipt_lines)


def plot_shap_waterfall(booster, features, feature_cols, save_path, txn_id=None):
    """Save a simple feature contribution plot (replacing SHAP waterfall)."""
    features_2d = np.array(features).reshape(1, -1)
    shap_values_with_base = booster.predict(features_2d, pred_contrib=True)[0]
    sv = shap_values_with_base[:-1]
    
    feature_importance = sorted(
        zip(feature_cols, sv),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:10]
    
    names = [x[0] for x in feature_importance]
    vals = [x[1] for x in feature_importance]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#ff8a80' if v > 0 else '#82b1ff' for v in vals]
    
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('SHAP Value (Contribution to Log Odds)')
    
    title = f"Top Feature Contributions — Transaction {txn_id}" if txn_id else "Top Feature Contributions"
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def explain_flagged_transactions(booster, calibrated_model, df, feature_cols, threshold=0.5, max_explain=100):
    """
    Explain all flagged (predicted fraud) transactions on a dataset.
    Returns list of explanation dicts.
    """
    X = df[feature_cols]
    y_pred = calibrated_model.predict_proba(X)[:, 1]
    flagged_mask = y_pred >= threshold
    flagged_indices = df.index[flagged_mask]

    logger.info(f"Flagged {flagged_mask.sum():,} transactions (threshold={threshold})")
    logger.info(f"Explaining top {min(max_explain, len(flagged_indices)):,}...")

    explanations = []
    for i, idx in enumerate(flagged_indices[:max_explain]):
        features = X.loc[idx].values
        expl = explain_transaction(booster, features, feature_cols)
        expl["transaction_id"] = int(df.loc[idx, "TransactionID"])
        expl["fraud_probability"] = float(y_pred[df.index.get_loc(idx)])
        explanations.append(expl)

        if (i + 1) % 50 == 0:
            logger.info(f"  Explained {i + 1}/{min(max_explain, len(flagged_indices))}")

    return explanations


if __name__ == "__main__":
    # Load model
    logger.info("Loading model...")
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm_model.txt"))
    calibrated_model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split_info = json.load(f)

    # Load test data
    logger.info("Loading test data...")
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "modeling_table.parquet"))
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    test = df.iloc[split_info["train_size"] + split_info["val_size"]:]

    # Explain flagged transactions
    explanations = explain_flagged_transactions(
        booster, calibrated_model, test, feature_cols, max_explain=50
    )

    # Save explanations
    out_path = os.path.join(MODEL_DIR, "explanations.json")
    with open(out_path, "w") as f:
        json.dump(explanations, f, indent=2)
    logger.info(f"\nSaved {len(explanations)} explanations to: {out_path}")

    # Plot a sample waterfall
    if len(explanations) > 0:
        sample_idx = test.index[0]
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_shap_waterfall(
            booster,
            test.loc[sample_idx, feature_cols].values,
            feature_cols,
            os.path.join(PLOTS_DIR, "shap_waterfall_sample.png"),
            txn_id=test.loc[sample_idx, "TransactionID"],
        )
        logger.info(f"Saved sample SHAP waterfall to: plots/shap_waterfall_sample.png")

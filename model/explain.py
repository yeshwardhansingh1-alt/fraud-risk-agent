"""
Days 12-13 — Explainability Layer [CORE].

SHAP TreeExplainer for feature-level attribution on flagged transactions.
Extract top 3-5 contributing features per flagged transaction.
Build "decision receipt" text generator that attaches to the agent's JSON output.
"""

import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import json
import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + "/..")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def create_explainer(booster, feature_cols):
    """Create a SHAP TreeExplainer for the LightGBM booster."""
    print("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(booster)
    return explainer


def explain_transaction(explainer, features, feature_cols, top_n=5):
    """
    Explain a single transaction.

    Args:
        explainer: SHAP TreeExplainer
        features: 1D array of feature values
        feature_cols: list of feature names
        top_n: number of top contributing features to return

    Returns:
        dict with SHAP values and top features
    """
    features_2d = np.array(features).reshape(1, -1)
    shap_values = explainer.shap_values(features_2d)

    # For binary classification, shap_values might be a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # Positive class (fraud)
    else:
        sv = shap_values[0]

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
        "base_value": round(float(explainer.expected_value if not isinstance(
            explainer.expected_value, list) else explainer.expected_value[1]), 4),
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


def plot_shap_waterfall(explainer, features, feature_cols, save_path, txn_id=None):
    """Save a SHAP waterfall plot for a single transaction."""
    features_2d = np.array(features).reshape(1, -1)

    # Create SHAP Explanation object
    shap_values = explainer.shap_values(features_2d)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        base_val = explainer.expected_value[1] if isinstance(
            explainer.expected_value, list) else explainer.expected_value
    else:
        sv = shap_values[0]
        base_val = explainer.expected_value

    explanation = shap.Explanation(
        values=sv,
        base_values=base_val,
        data=features.flatten(),
        feature_names=feature_cols,
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    title = f"SHAP Waterfall — Transaction {txn_id}" if txn_id else "SHAP Waterfall"
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
    explainer = create_explainer(booster, feature_cols)

    X = df[feature_cols]
    y_pred = calibrated_model.predict_proba(X)[:, 1]
    flagged_mask = y_pred >= threshold
    flagged_indices = df.index[flagged_mask]

    print(f"Flagged {flagged_mask.sum():,} transactions (threshold={threshold})")
    print(f"Explaining top {min(max_explain, len(flagged_indices)):,}...")

    explanations = []
    for i, idx in enumerate(flagged_indices[:max_explain]):
        features = X.loc[idx].values
        expl = explain_transaction(explainer, features, feature_cols)
        expl["transaction_id"] = int(df.loc[idx, "TransactionID"])
        expl["fraud_probability"] = float(y_pred[df.index.get_loc(idx)])
        explanations.append(expl)

        if (i + 1) % 50 == 0:
            print(f"  Explained {i + 1}/{min(max_explain, len(flagged_indices))}")

    return explanations


if __name__ == "__main__":
    # Load model
    print("Loading model...")
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm_model.txt"))
    calibrated_model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split_info = json.load(f)

    # Load test data
    print("Loading test data...")
    df = pd.read_csv(os.path.join(FEATURES_DIR, "modeling_table.csv"))
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
    print(f"\nSaved {len(explanations)} explanations to: {out_path}")

    # Plot a sample waterfall
    if len(explanations) > 0:
        explainer = create_explainer(booster, feature_cols)
        sample_idx = test.index[0]
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_shap_waterfall(
            explainer,
            test.loc[sample_idx, feature_cols].values,
            feature_cols,
            os.path.join(PLOTS_DIR, "shap_waterfall_sample.png"),
            txn_id=test.loc[sample_idx, "TransactionID"],
        )
        print(f"Saved sample SHAP waterfall to: plots/shap_waterfall_sample.png")

    print("\nDays 12-13 complete.")

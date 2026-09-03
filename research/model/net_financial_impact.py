"""
Day 16 — Net Financial Impact [STRETCH].

NFI = prevented fraud value – false positive friction costs – chargeback fees.
Compare rule-engine baseline (Day 5) vs ML detector/agent on the same test set.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
import os
import sys

from model.cost_model import COST_CONFIG
from agent.expected_loss import argmin_action, cfraud, clegit

MODEL_DIR = os.path.dirname(__file__)
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def compute_nfi(y_true, y_pred, amounts, config=None):
    """
    Compute Net Financial Impact.

    NFI = prevented_fraud_value - FP_friction_costs - chargeback_fees

    Args:
        y_true: actual fraud labels
        y_pred: binary predictions (0/1)
        amounts: transaction amounts
        config: cost config dict

    Returns:
        dict with NFI breakdown
    """
    if config is None:
        config = COST_CONFIG

    tp = (y_pred == 1) & (y_true == 1)  # Correctly blocked fraud
    fp = (y_pred == 1) & (y_true == 0)  # Incorrectly blocked legit
    fn = (y_pred == 0) & (y_true == 1)  # Missed fraud

    # Prevented fraud value (what we saved by blocking)
    prevented_fraud_value = (amounts[tp] * config["fraud_loss_fraction"]).sum()
    prevented_chargebacks = tp.sum() * config["chargeback_fee"]

    # False positive costs
    fp_friction = fp.sum() * config["false_positive_friction_cost"]
    fp_lost_revenue = (amounts[fp] * config["lost_revenue_fraction"]).sum()

    # Missed fraud chargebacks
    missed_chargebacks = fn.sum() * config["chargeback_fee"]
    missed_fraud_loss = (amounts[fn] * config["fraud_loss_fraction"]).sum()

    nfi = (prevented_fraud_value + prevented_chargebacks) - (fp_friction + fp_lost_revenue)

    nfi = (prevented_fraud_value + prevented_chargebacks) - (fp_friction + fp_lost_revenue)

    return {
        "prevented_fraud_value": float(prevented_fraud_value),
        "prevented_chargebacks": float(prevented_chargebacks),
        "fp_friction_cost": float(fp_friction),
        "fp_lost_revenue": float(fp_lost_revenue),
        "missed_chargebacks": float(missed_chargebacks),
        "missed_fraud_loss": float(missed_fraud_loss),
        "nfi": float(nfi),
        "true_positives": int(tp.sum()),
        "false_positives": int(fp.sum()),
        "false_negatives": int(fn.sum()),
        "precision": float(tp.sum() / (tp.sum() + fp.sum())) if (tp.sum() + fp.sum()) > 0 else 0,
        "recall": float(tp.sum() / (tp.sum() + fn.sum())) if (tp.sum() + fn.sum()) > 0 else 0,
    }


def compute_agent_nfi(y_true, y_pred_proba, amounts):
    """
    Compute Net Financial Impact using the full 4-action Decision Agent policy.
    NFI = (Loss under Approve-All) - (Loss under Agent Policy)
    """
    total_approve_all_loss = 0.0
    total_agent_loss = 0.0
    
    prevented_fraud_value = 0.0
    fp_cost = 0.0

    actions = []
    
    for p_fraud, y, amt in zip(y_pred_proba, y_true, amounts):
        # Baseline loss (Approve everything)
        baseline_loss = cfraud("approve", amt) if y == 1 else clegit("approve", amt)
        total_approve_all_loss += baseline_loss
        
        # Agent loss
        best_action, _ = argmin_action(p_fraud, amt)
        actions.append(best_action)
        agent_loss = cfraud(best_action, amt) if y == 1 else clegit(best_action, amt)
        total_agent_loss += agent_loss
        
        # Approximate metrics for plotting parity
        savings = baseline_loss - agent_loss
        if y == 1 and savings > 0:
            prevented_fraud_value += savings
        elif y == 0 and savings < 0:
            fp_cost += abs(savings)
            
    nfi = total_approve_all_loss - total_agent_loss
    
    return {
        "prevented_fraud_value": float(prevented_fraud_value),
        "prevented_chargebacks": 0.0, # Merged into value for agent
        "fp_friction_cost": float(fp_cost),
        "fp_lost_revenue": 0.0,
        "nfi": float(nfi),
        "actions": pd.Series(actions).value_counts().to_dict()
    }


def plot_nfi_comparison(baseline_nfi, ml_nfi, save_path):
    """Bar chart comparing NFI: rule-engine baseline vs. ML detector."""
    categories = [
        "Prevented\nFraud Value",
        "Prevented\nChargebacks",
        "FP Friction\nCost",
        "FP Lost\nRevenue",
        "Net Financial\nImpact",
    ]

    baseline_vals = [
        baseline_nfi["prevented_fraud_value"],
        baseline_nfi["prevented_chargebacks"],
        -baseline_nfi["fp_friction_cost"],
        -baseline_nfi["fp_lost_revenue"],
        baseline_nfi["nfi"],
    ]

    ml_vals = [
        ml_nfi["prevented_fraud_value"],
        ml_nfi["prevented_chargebacks"],
        -ml_nfi["fp_friction_cost"],
        -ml_nfi["fp_lost_revenue"],
        ml_nfi["nfi"],
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Rule Engine (Baseline)",
                   color="#FF9800", alpha=0.8)
    bars2 = ax.bar(x + width/2, ml_vals, width, label="ML Detector (LightGBM)",
                   color="#2196F3", alpha=0.8)

    ax.set_ylabel("Dollar Value ($)")
    ax.set_title("Net Financial Impact: Rule Engine vs. ML Detector")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"${abs(height):,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"${abs(height):,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"  Saved NFI comparison chart: {save_path}")


if __name__ == "__main__":
    # Load data
    logger.info("Loading test data...")
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "modeling_table.parquet"))
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split_info = json.load(f)
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    test = df.iloc[split_info["train_size"] + split_info["val_size"]:]
    y_true = test["isFraud"].values
    amounts = test["TransactionAmt"].values

    # --- Rule Engine Baseline (Day 5) ---
    logger.info("\nLoading rule engine outputs...")
    rule_outputs = pd.read_csv(os.path.join(FEATURES_DIR, "rule_engine_outputs.csv"))
    # Merge on TransactionID to align with test set
    test_with_rules = test.merge(rule_outputs[["TransactionID", "rule_any_fired"]],
                                  on="TransactionID", how="left")
    rule_preds = test_with_rules["rule_any_fired"].fillna(0).astype(int).values
    baseline_nfi = compute_nfi(y_true, rule_preds, amounts)

    logger.info(f"\nRule Engine NFI: ${baseline_nfi['nfi']:,.2f}")
    logger.info(f"  Precision: {baseline_nfi['precision']:.4f}, Recall: {baseline_nfi['recall']:.4f}")

    logger.info("\nLoading ML model predictions and applying Agent Logic...")
    calibrated_model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))
    y_pred_proba = calibrated_model.predict_proba(test[feature_cols])[:, 1]
    
    agent_nfi = compute_agent_nfi(y_true, y_pred_proba, amounts)

    logger.info(f"\nDecision Agent NFI: ${agent_nfi['nfi']:,.2f}")
    logger.info(f"  Actions taken: {agent_nfi['actions']}")

    # --- Comparison ---
    improvement = agent_nfi["nfi"] - baseline_nfi["nfi"]
    logger.info(f"\n{'='*60}")
    logger.info(f"NFI Improvement (Agent over Rules): ${improvement:,.2f}")
    logger.info(f"{'='*60}")

    # Plot comparison
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_nfi_comparison(baseline_nfi, agent_nfi, os.path.join(PLOTS_DIR, "nfi_comparison.png"))

    # Save results
    results = {
        "rule_engine": baseline_nfi,
        "ml_detector": agent_nfi,
        "improvement": float(improvement),
    }
    with open(os.path.join(MODEL_DIR, "nfi_results.json"), "w") as f:
        json.dump(results, f, indent=2)



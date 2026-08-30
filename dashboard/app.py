"""
Days 22-25 — Streamlit Dashboard.

- Replay view: step through test-set transactions one at a time or on a timer
- Live agent decision per transaction (action + probability)
- Running $-saved counter (agent NFI vs. baseline, live-updating)
- Explainability panel: SHAP waterfall on click
- Metrics panel: calibration curve, ROC/PR curves, latency histogram
"""

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import joblib
import json
import os
import sys
import time

from model.cost_model import COST_CONFIG

# Add project paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
from agent.expected_loss import argmin_action

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# --- Page Config ---
st.set_page_config(
    page_title="Fraud Risk Agent — Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_models():
    """Load all model artifacts (cached)."""
    calibrated_model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm_model.txt"))

    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split_info = json.load(f)

    return calibrated_model, booster, feature_cols, split_info


@st.cache_data
def load_test_data(_split_info, feature_cols):
    """Load test set data (cached)."""
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "modeling_table.parquet"))
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    test = df.iloc[_split_info["train_size"] + _split_info["val_size"]:].copy()

    return test


def make_decision_for_row(row, calibrated_model, feature_cols):
    """Get fraud decision for a single row."""
    features = row[feature_cols].values.reshape(1, -1)
    p_fraud = float(calibrated_model.predict_proba(features)[0, 1])
    best_action, losses = argmin_action(p_fraud, row["TransactionAmt"])
    return p_fraud, best_action, losses


def render_sidebar():
    """Render sidebar with navigation."""
    st.sidebar.title("🛡️ Fraud Risk Agent")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["🔄 Transaction Replay", "📊 Metrics Dashboard", "📋 About"],
    )
    return page


def render_replay_page(calibrated_model, booster, feature_cols, split_info, test):
    """Day 22-24: Transaction replay with live decisions."""
    st.title("🔄 Transaction Replay")
    st.markdown("Step through test-set transactions and see live agent decisions.")

    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        txn_index = st.slider("Transaction #", 0, len(test) - 1, 0)
    with col2:
        auto_play = st.checkbox("Auto-play")
    with col3:
        speed = st.slider("Speed (sec)", 0.5, 3.0, 1.0)

    row = test.iloc[txn_index]

    # Get decision
    p_fraud, action, losses = make_decision_for_row(row, calibrated_model, feature_cols)

    # --- Running $-saved counter ---
    # Calculate cumulative NFI up to this transaction
    if "cumulative_nfi" not in st.session_state:
        st.session_state.cumulative_nfi = 0.0
        st.session_state.cumulative_blocked_fraud = 0.0
        st.session_state.cumulative_fp_cost = 0.0
        st.session_state.seen_txns = set()

    is_fraud = row["isFraud"] == 1
    amt = row["TransactionAmt"]

    if txn_index not in st.session_state.seen_txns:
        st.session_state.seen_txns.add(txn_index)
        if action in ("block", "step_up"):
            if is_fraud:
                st.session_state.cumulative_blocked_fraud += amt * COST_CONFIG.get("fraud_loss_fraction", 1.0)
            else:
                st.session_state.cumulative_fp_cost += COST_CONFIG.get("false_positive_friction_cost", 25.0) + amt * COST_CONFIG.get("lost_revenue_fraction", 0.8)
        st.session_state.cumulative_nfi = st.session_state.cumulative_blocked_fraud - st.session_state.cumulative_fp_cost

    # Display metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        action_color = {"approve": "🟢", "step_up": "🟡", "block": "🔴", "auto_dispute": "🟠"}
        st.metric("Action", f"{action_color.get(action, '⚪')} {action.upper()}")
    with m2:
        st.metric("Fraud Probability", f"{p_fraud:.4f}")
    with m3:
        st.metric("Transaction Amount", f"${amt:,.2f}")
    with m4:
        st.metric("Net $-Saved", f"${st.session_state.cumulative_nfi:,.2f}",
                   delta=f"${st.session_state.cumulative_blocked_fraud:,.0f} blocked")

    # Transaction details
    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Transaction Details")
        actual_label = "🚨 FRAUD" if is_fraud else "✅ LEGITIMATE"
        st.markdown(f"**Actual:** {actual_label}")
        st.markdown(f"**Transaction ID:** {row['TransactionID']:.0f}")
        st.markdown(f"**Amount:** ${amt:,.2f}")

        st.markdown("**Expected Losses by Action:**")
        for a, l in sorted(losses.items(), key=lambda x: x[1]):
            marker = " ← chosen" if a == action else ""
            st.text(f"  {a:15s}: ${l:>8.2f}{marker}")

    with col_right:
        st.subheader("SHAP Explanation")
        # Compute SHAP values via LightGBM native
        features = row[feature_cols].values.reshape(1, -1)
        shap_values_with_base = booster.predict(features, pred_contrib=True)[0]
        sv = shap_values_with_base[:-1]

        # Top 5 features
        sorted_feats = sorted(zip(feature_cols, sv), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feat_name, shap_val in sorted_feats:
            direction = "↑" if shap_val > 0 else "↓"
            st.text(f"  {direction} {feat_name}: SHAP={shap_val:+.4f}")

        # Simple bar chart (replacing waterfall)
        top_10 = sorted(zip(feature_cols, sv), key=lambda x: abs(x[1]), reverse=True)[:10]
        names = [x[0] for x in top_10]
        vals = [x[1] for x in top_10]
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#ff8a80' if v > 0 else '#82b1ff' for v in vals]
        y_pos = np.arange(len(names))
        ax.barh(y_pos, vals, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('SHAP Value')
        st.pyplot(fig)
        plt.close()

    # Auto-play
    if auto_play:
        time.sleep(speed)
        st.rerun()


def render_metrics_page(calibrated_model, feature_cols, split_info, test):
    """Day 25: Metrics dashboard."""
    st.title("📊 Metrics Dashboard")

    y_true = test["isFraud"].values
    y_pred_proba = calibrated_model.predict_proba(test[feature_cols])[:, 1]

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Calibration", "ROC / PR Curves", "Results Summary"])

    with tab1:
        st.subheader("Calibration Curve (Reliability Diagram)")
        frac_pos, mean_pred = calibration_curve(y_true, y_pred_proba, n_bins=20, strategy="quantile")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(mean_pred, frac_pos, "s-", color="#2196F3", label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Fraud Rate")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {roc_auc:.4f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = average_precision_score(y_true, y_pred_proba)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(recall, precision, color="#4CAF50", lw=2, label=f"AP = {pr_auc:.4f}")
            ax.axhline(y_true.mean(), color="gray", linestyle="--", label="Baseline")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend()
            st.pyplot(fig)
            plt.close()

    with tab3:
        st.subheader("Key Results")
        # Load saved results if available
        results_path = os.path.join(MODEL_DIR, "backtest_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)

            cm = results.get("classification_metrics", {})
            cost = results.get("cost_metrics", {})

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Precision", f"{cm.get('precision', 0):.4f}")
            col2.metric("Recall", f"{cm.get('recall', 0):.4f}")
            col3.metric("ROC-AUC", f"{cm.get('roc_auc', 0):.4f}")
            col4.metric("PR-AUC", f"{cm.get('pr_auc', 0):.4f}")

            st.markdown("---")
            col1, col2 = st.columns(2)
            col1.metric("FP Cost / 1,000 txn", f"${cost.get('false_positive_cost_per_1000_txn', 0):,.2f}")
            col2.metric("Missed Fraud / 1,000 txn", f"${cost.get('missed_fraud_cost_per_1000_txn', 0):,.2f}")
        else:
            st.info("Run model/evaluate.py first to generate backtest results.")

        # Load latency results
        latency_path = os.path.join(MODEL_DIR, "latency_results.json")
        if os.path.exists(latency_path):
            with open(latency_path) as f:
                lat = json.load(f)
            st.markdown("---")
            st.subheader("Latency")
            col1, col2, col3 = st.columns(3)
            col1.metric("p50", f"{lat.get('p50_ms', 0):.1f} ms")
            col2.metric("p95", f"{lat.get('p95_ms', 0):.1f} ms")
            col3.metric("p99", f"{lat.get('p99_ms', 0):.1f} ms")


def render_about_page():
    """About page."""
    st.title("📋 About")
    st.markdown("""
    ## Fraud Risk Agent

    A cost-sensitive fraud-spike detector built for Razorpay's AI Buildathon
    (Track 02: AI Risk Manager).

    ### Architecture
    - **Model**: Calibrated LightGBM (isotonic regression)
    - **Features**: Velocity, entity graph, behavioral, time-based
    - **Decision**: Expected-loss minimization over 4 actions
    - **Explainability**: SHAP TreeExplainer per flagged transaction

    ### Defense-Only
    This system detects and explains fraud. It does not help commit it.
    Every flagged transaction ships with a SHAP-based reason code explaining
    *why* it was flagged, not just a score.
    """)


# --- Main ---
def main():
    page = render_sidebar()

    try:
        calibrated_model, booster, feature_cols, split_info = load_models()
        test = load_test_data(split_info, feature_cols)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure you've run the training pipeline first (Days 1-7).")
        return

    if "Replay" in page:
        render_replay_page(calibrated_model, booster, feature_cols, split_info, test)
    elif "Metrics" in page:
        render_metrics_page(calibrated_model, feature_cols, split_info, test)
    elif "About" in page:
        render_about_page()


if __name__ == "__main__":
    main()

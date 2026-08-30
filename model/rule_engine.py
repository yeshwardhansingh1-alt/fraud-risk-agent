"""
Day 5 — Baseline Rule Engine [STRETCH].

Hand-written fraud rules: Approve/Block only, no probabilities.
Used later (Day 16) as the baseline for Net Financial Impact comparison.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import os


# --- Rule Definitions (Vectorized) ---
# Each rule takes a DataFrame and returns a boolean Series.

def rule_high_amount_new_device(df):
    """Block if amount > $500 AND device is new (no prior transactions)."""
    amt = df.get("TransactionAmt", pd.Series(0, index=df.index))
    dev = df.get("device_txn_count_24hr", pd.Series(0, index=df.index))
    return (amt > 500) & (dev == 0)

def rule_velocity_spike(df):
    """Block if card has > 5 transactions in the last 10 minutes."""
    vel = df.get("card1_txn_count_10min", pd.Series(0, index=df.index))
    return vel > 5

def rule_impossible_travel(df):
    """Block if impossible travel detected."""
    trav = df.get("impossible_travel", pd.Series(0, index=df.index))
    return trav == 1

def rule_high_amount_zscore(df):
    """Block if amount z-score > 3 (very unusual for this card)."""
    zscore = df.get("amount_zscore_card", pd.Series(0, index=df.index))
    return zscore.abs() > 3

def rule_many_cards_on_device(df):
    """Block if > 5 distinct cards share the same device."""
    cards = df.get("cards_sharing_device", pd.Series(0, index=df.index))
    return cards > 5

def rule_late_night_high_amount(df):
    """Block if transaction is between 1-5 AM AND amount > $300."""
    hour = df.get("hour_of_day", pd.Series(12, index=df.index))
    amt = df.get("TransactionAmt", pd.Series(0, index=df.index))
    return (hour >= 1) & (hour <= 5) & (amt > 300)

ALL_RULES = [
    ("high_amount_new_device", rule_high_amount_new_device),
    ("velocity_spike", rule_velocity_spike),
    ("impossible_travel", rule_impossible_travel),
    ("high_amount_zscore", rule_high_amount_zscore),
    ("many_cards_on_device", rule_many_cards_on_device),
    ("late_night_high_amount", rule_late_night_high_amount),
]


def score_with_rules(df):
    """
    Score every transaction with the rule engine (Vectorized).
    Returns df with 'rule_decision' column: 'Approve' or 'Block'.
    Also adds individual rule columns for analysis.
    """
    logger.info("Scoring with rule engine (vectorized)...")
    for rule_name, rule_fn in ALL_RULES:
        df[f"rule_{rule_name}"] = rule_fn(df).astype(int)

    # Block if ANY rule fires
    rule_cols = [f"rule_{name}" for name, _ in ALL_RULES]
    df["rule_any_fired"] = df[rule_cols].max(axis=1)
    df["rule_decision"] = df["rule_any_fired"].map({0: "Approve", 1: "Block"})

    # Stats
    n_blocked = (df["rule_decision"] == "Block").sum()
    n_total = len(df)
    logger.info(f"  Blocked: {n_blocked:,} / {n_total:,} ({n_blocked/n_total:.2%})")

    # Among blocked, how many are actual fraud?
    blocked = df[df["rule_decision"] == "Block"]
    if "isFraud" in df.columns:
        precision = blocked["isFraud"].mean() if len(blocked) > 0 else 0
        recall = blocked["isFraud"].sum() / df["isFraud"].sum() if df["isFraud"].sum() > 0 else 0
        logger.info(f"  Rule precision: {precision:.4f}")
        logger.info(f"  Rule recall:    {recall:.4f}")

        # Per-rule stats
        logger.info("\n  Per-rule breakdown:")
        for rule_name, _ in ALL_RULES:
            col = f"rule_{rule_name}"
            fired = df[df[col] == 1]
            n_fired = len(fired)
            if n_fired > 0:
                prec = fired["isFraud"].mean()
                rec = fired["isFraud"].sum() / df["isFraud"].sum()
                logger.info(f"    {rule_name:30s}  fired={n_fired:>6,}  prec={prec:.4f}  recall={rec:.4f}")

    return df


if __name__ == "__main__":
    FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")

    logger.info("Loading modeling table...")
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "modeling_table.parquet"))

    df = score_with_rules(df)

    # Save rule engine outputs for Day 16 NFI comparison
    out_path = os.path.join(FEATURES_DIR, "rule_engine_outputs.csv")
    rule_cols = ["TransactionID", "rule_decision", "rule_any_fired"] + \
                [f"rule_{name}" for name, _ in ALL_RULES]
    df[rule_cols].to_csv(out_path, index=False)
    logger.info(f"\nSaved rule engine outputs to: {out_path}")

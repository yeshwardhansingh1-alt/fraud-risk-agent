"""
Day 5 — Baseline Rule Engine [STRETCH].

Hand-written fraud rules: Approve/Block only, no probabilities.
Used later (Day 16) as the baseline for Net Financial Impact comparison.
"""

import pandas as pd
import numpy as np
import os


# --- Rule Definitions ---
# Each rule returns True if the transaction should be BLOCKED.

def rule_high_amount_new_device(row):
    """Block if amount > $500 AND device is new (no prior transactions)."""
    return row.get("TransactionAmt", 0) > 500 and row.get("device_txn_count_24hr", 0) == 0


def rule_velocity_spike(row):
    """Block if card has > 5 transactions in the last 10 minutes."""
    return row.get("card1_txn_count_10min", 0) > 5


def rule_impossible_travel(row):
    """Block if impossible travel detected."""
    return row.get("impossible_travel", 0) == 1


def rule_high_amount_zscore(row):
    """Block if amount z-score > 3 (very unusual for this card)."""
    return abs(row.get("amount_zscore_card", 0)) > 3


def rule_many_cards_on_device(row):
    """Block if > 5 distinct cards share the same device."""
    return row.get("cards_sharing_device", 0) > 5


def rule_late_night_high_amount(row):
    """Block if transaction is between 1-5 AM AND amount > $300."""
    hour = row.get("hour_of_day", 12)
    amt = row.get("TransactionAmt", 0)
    return 1 <= hour <= 5 and amt > 300


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
    Score every transaction with the rule engine.
    Returns df with 'rule_decision' column: 'Approve' or 'Block'.
    Also adds individual rule columns for analysis.
    """
    print("Scoring with rule engine...")
    for rule_name, rule_fn in ALL_RULES:
        df[f"rule_{rule_name}"] = df.apply(rule_fn, axis=1).astype(int)

    # Block if ANY rule fires
    rule_cols = [f"rule_{name}" for name, _ in ALL_RULES]
    df["rule_any_fired"] = df[rule_cols].max(axis=1)
    df["rule_decision"] = df["rule_any_fired"].map({0: "Approve", 1: "Block"})

    # Stats
    n_blocked = (df["rule_decision"] == "Block").sum()
    n_total = len(df)
    print(f"  Blocked: {n_blocked:,} / {n_total:,} ({n_blocked/n_total:.2%})")

    # Among blocked, how many are actual fraud?
    blocked = df[df["rule_decision"] == "Block"]
    if "isFraud" in df.columns:
        precision = blocked["isFraud"].mean() if len(blocked) > 0 else 0
        recall = blocked["isFraud"].sum() / df["isFraud"].sum() if df["isFraud"].sum() > 0 else 0
        print(f"  Rule precision: {precision:.4f}")
        print(f"  Rule recall:    {recall:.4f}")

        # Per-rule stats
        print("\n  Per-rule breakdown:")
        for rule_name, _ in ALL_RULES:
            col = f"rule_{rule_name}"
            fired = df[df[col] == 1]
            n_fired = len(fired)
            if n_fired > 0:
                prec = fired["isFraud"].mean()
                rec = fired["isFraud"].sum() / df["isFraud"].sum()
                print(f"    {rule_name:30s}  fired={n_fired:>6,}  prec={prec:.4f}  recall={rec:.4f}")

    return df


if __name__ == "__main__":
    FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")

    print("Loading modeling table...")
    df = pd.read_csv(os.path.join(FEATURES_DIR, "modeling_table.csv"))

    df = score_with_rules(df)

    # Save rule engine outputs for Day 16 NFI comparison
    out_path = os.path.join(FEATURES_DIR, "rule_engine_outputs.csv")
    rule_cols = ["TransactionID", "rule_decision", "rule_any_fired"] + \
                [f"rule_{name}" for name, _ in ALL_RULES]
    df[rule_cols].to_csv(out_path, index=False)
    print(f"\nSaved rule engine outputs to: {out_path}")
    print("Day 5 complete.")

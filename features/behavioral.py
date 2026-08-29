"""
Day 4 — Behavioral Features.

Amount z-score vs. card's own historical mean/std (expanding window).
"Impossible travel" flag using addr columns as a geo proxy.
"""

import pandas as pd
import numpy as np


def build_behavioral_features(df):
    """Build behavioral features. Expects df sorted by TransactionDT ascending."""
    df = df.sort_values("TransactionDT").copy()

    # Amount z-score vs card's own history (expanding, shifted to avoid leakage)
    print("  Building amount_zscore_card...")
    card_groups = df.groupby("card1")["TransactionAmt"]
    expanding_mean = card_groups.apply(lambda x: x.expanding().mean().shift(1)).droplevel(0).sort_index()
    expanding_std = card_groups.apply(lambda x: x.expanding().std().shift(1)).droplevel(0).sort_index()
    df["card_amt_expanding_mean"] = expanding_mean
    df["card_amt_expanding_std"] = expanding_std
    df["amount_zscore_card"] = (
        (df["TransactionAmt"] - df["card_amt_expanding_mean"])
        / df["card_amt_expanding_std"].replace(0, np.nan)
    ).fillna(0)
    df.drop(columns=["card_amt_expanding_mean", "card_amt_expanding_std"], inplace=True)

    # "Impossible travel" flag
    print("  Building impossible_travel flag...")
    df["_prev_addr1"] = df.groupby("card1")["addr1"].shift(1)
    df["_prev_addr2"] = df.groupby("card1")["addr2"].shift(1)
    df["_time_since_last"] = df.groupby("card1")["TransactionDT"].diff()

    addr_changed = (df["addr1"] != df["_prev_addr1"]) | (df["addr2"] != df["_prev_addr2"])
    short_gap = df["_time_since_last"] < 3600
    has_prev = df["_prev_addr1"].notna()
    df["impossible_travel"] = (addr_changed & short_gap & has_prev).astype(int)
    df.drop(columns=["_prev_addr1", "_prev_addr2", "_time_since_last"], inplace=True)

    print(f"  Impossible travel flags: {df['impossible_travel'].sum():,} ({df['impossible_travel'].mean():.4%})")
    return df

"""
Day 3 — Velocity Features.

Rolling counts of transactions per card/device in 10min / 1hr / 24hr windows.
Time-since-last-transaction per card. Includes future data leak sanity check.
"""

import pandas as pd
import numpy as np
import os

WINDOWS = {"10min": 600, "1hr": 3600, "24hr": 86400}


def _rolling_count(df, group_col, time_col, window_seconds, feature_name):
    """Count transactions for same entity within past window_seconds. No future leakage."""
    counts = []
    for _, group in df.groupby(group_col):
        times = group[time_col].values
        cnts = np.zeros(len(times), dtype=np.int32)
        for i in range(len(times)):
            window_start = times[i] - window_seconds
            cnts[i] = np.sum(times[:i] >= window_start)
        counts.append(pd.Series(cnts, index=group.index, name=feature_name))
    return pd.concat(counts).sort_index()


def build_velocity_features(df):
    """Build all velocity features. Expects df sorted by TransactionDT ascending."""
    df = df.sort_values("TransactionDT").copy()

    for window_name, window_sec in WINDOWS.items():
        col = f"card1_txn_count_{window_name}"
        print(f"  Building {col}...")
        df[col] = _rolling_count(df, "card1", "TransactionDT", window_sec, col)

    if "DeviceInfo" in df.columns:
        df["_device"] = df["DeviceInfo"].fillna("UNKNOWN")
        for window_name, window_sec in WINDOWS.items():
            col = f"device_txn_count_{window_name}"
            print(f"  Building {col}...")
            df[col] = _rolling_count(df, "_device", "TransactionDT", window_sec, col)
        df.drop(columns=["_device"], inplace=True)

    print("  Building time_since_last_txn_card...")
    df["time_since_last_txn_card"] = df.groupby("card1")["TransactionDT"].diff().fillna(-1)

    return df


def sanity_check_no_future_leakage(df):
    """Confirm velocity features don't look into future rows."""
    print("\n  Sanity check: no future data leakage...")
    first_txn_idx = df.groupby("card1")["TransactionDT"].idxmin()
    first_txns = df.loc[first_txn_idx]
    velocity_cols = [c for c in df.columns if "txn_count" in c and "card1" in c]
    all_ok = True
    for col in velocity_cols:
        if first_txns[col].max() > 0:
            print(f"    FAIL: {col} has non-zero on first transactions!")
            all_ok = False
        else:
            print(f"    OK: {col}")
    return all_ok


if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    txn = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"))
    ident = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"))
    df = txn.merge(ident, on="TransactionID", how="left")
    print("Building velocity features...")
    df = build_velocity_features(df)
    sanity_check_no_future_leakage(df)
    print("Day 3 complete.")

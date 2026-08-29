"""
Day 1 — Load IEEE-CIS Fraud Detection data, merge tables, inspect basics.

Expects the following CSVs in data/:
  - train_transaction.csv
  - train_identity.csv

Outputs a summary of shape, nulls, and fraud rate to stdout.
"""

import pandas as pd
import numpy as np
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_and_merge():
    """Load transaction + identity tables and merge on TransactionID."""
    txn_path = os.path.join(DATA_DIR, "train_transaction.csv")
    id_path = os.path.join(DATA_DIR, "train_identity.csv")

    for p in [txn_path, id_path]:
        if not os.path.exists(p):
            print(f"ERROR: file not found -> {p}")
            print("Please download the IEEE-CIS dataset from Kaggle first:")
            print("  kaggle competitions download -c ieee-fraud-detection -p data/")
            sys.exit(1)

    print("Loading train_transaction.csv ...")
    txn = pd.read_csv(txn_path)
    print(f"  -> shape: {txn.shape}")

    print("Loading train_identity.csv ...")
    ident = pd.read_csv(id_path)
    print(f"  -> shape: {ident.shape}")

    print("Merging on TransactionID (left join) ...")
    df = txn.merge(ident, on="TransactionID", how="left")
    print(f"  -> merged shape: {df.shape}")

    return df


def inspect(df):
    """Print shape, null summary, fraud rate, and basic stats."""
    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    # Shape
    print(f"\nRows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    # Fraud rate
    fraud_rate = df["isFraud"].mean()
    fraud_count = df["isFraud"].sum()
    print(f"\nFraud rate: {fraud_rate:.4%}  ({fraud_count:,} / {len(df):,})")

    # Null summary — columns with most nulls
    null_pct = df.isnull().mean().sort_values(ascending=False)
    print(f"\nColumns with >50% nulls: {(null_pct > 0.50).sum()}")
    print(f"Columns with >90% nulls: {(null_pct > 0.90).sum()}")
    print(f"Columns with zero nulls: {(null_pct == 0).sum()}")

    print("\nTop 15 columns by null %:")
    for col, pct in null_pct.head(15).items():
        print(f"  {col:40s} {pct:6.2%}")

    # Data types
    print(f"\nData types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {str(dtype):20s} {count}")

    # TransactionAmt quick stats
    print(f"\nTransactionAmt stats:")
    print(df["TransactionAmt"].describe().to_string())

    print("\n" + "=" * 60)
    print("Day 1 inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    df = load_and_merge()
    inspect(df)

"""
Day 1 — Load IEEE-CIS Fraud Detection data, merge tables, inspect basics.

Expects the following CSVs in data/:
  - train_transaction.csv
  - train_identity.csv

Outputs a summary of shape, nulls, and fraud rate to stdout.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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
            logger.info(f"ERROR: file not found -> {p}")
            logger.info("Please download the IEEE-CIS dataset from Kaggle first:")
            logger.info("  kaggle competitions download -c ieee-fraud-detection -p data/")
            sys.exit(1)

    logger.info("Loading train_transaction.csv ...")
    txn = pd.read_csv(txn_path)
    logger.info(f"  -> shape: {txn.shape}")

    logger.info("Loading train_identity.csv ...")
    ident = pd.read_csv(id_path)
    logger.info(f"  -> shape: {ident.shape}")

    logger.info("Merging on TransactionID (left join) ...")
    df = txn.merge(ident, on="TransactionID", how="left")
    logger.info(f"  -> merged shape: {df.shape}")

    return df


def inspect(df):
    """Print shape, null summary, fraud rate, and basic stats."""
    logger.info("\n" + "=" * 60)
    logger.info("DATASET INSPECTION")
    logger.info("=" * 60)

    # Shape
    logger.info(f"\nRows:    {df.shape[0]:,}")
    logger.info(f"Columns: {df.shape[1]:,}")

    # Fraud rate
    fraud_rate = df["isFraud"].mean()
    fraud_count = df["isFraud"].sum()
    logger.info(f"\nFraud rate: {fraud_rate:.4%}  ({fraud_count:,} / {len(df):,})")

    # Null summary — columns with most nulls
    null_pct = df.isnull().mean().sort_values(ascending=False)
    logger.info(f"\nColumns with >50% nulls: {(null_pct > 0.50).sum()}")
    logger.info(f"Columns with >90% nulls: {(null_pct > 0.90).sum()}")
    logger.info(f"Columns with zero nulls: {(null_pct == 0).sum()}")

    logger.info("\nTop 15 columns by null %:")
    for col, pct in null_pct.head(15).items():
        logger.info(f"  {col:40s} {pct:6.2%}")

    # Data types
    logger.info(f"\nData types:")
    for dtype, count in df.dtypes.value_counts().items():
        logger.info(f"  {str(dtype):20s} {count}")

    # TransactionAmt quick stats
    logger.info(f"\nTransactionAmt stats:")
    logger.info(df["TransactionAmt"].describe().to_string())

    logger.info("\n" + "=" * 60)

    logger.info("=" * 60)


if __name__ == "__main__":
    df = load_and_merge()
    inspect(df)

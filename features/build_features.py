"""
Day 4 — Build Features: merge all engineered features into one modeling table.

Orchestrates velocity, entity graph, and behavioral feature pipelines,
then produces a single features/modeling_table.parquet ready for training.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import os
import sys

from features.velocity import build_velocity_features, sanity_check_no_future_leakage
from features.entity_graph import build_graph_features_fast
from features.behavioral import build_behavioral_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES_DIR = os.path.dirname(__file__)


def build_all_features():
    """Load raw data, build all features, merge into one table."""

    import pandera as pa
    from pandera.typing import DataFrame, Series

    # --- Load raw data ---
    logger.info("Loading raw data...")
    txn = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"))
    ident = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"))
    
    # Define and apply raw data schema checks
    logger.info("Validating raw data schema with Pandera...")
    txn_schema = pa.DataFrameSchema({
        "TransactionID": pa.Column(int, unique=True, coerce=True),
        "TransactionDT": pa.Column(int, coerce=True),
        "TransactionAmt": pa.Column(float, pa.Check.ge(0), coerce=True),
        "isFraud": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
        "card1": pa.Column(int, coerce=True)
    })
    
    txn = txn_schema.validate(txn)
    
    df = txn.merge(ident, on="TransactionID", how="left")
    logger.info(f"  Raw merged shape: {df.shape}")

    # Sort by time — critical for all feature engineering
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    
    # (Removed 200k sandbox cap — processing full dataset)

    # --- Velocity features (Day 3) ---
    logger.info("\n--- Velocity Features ---")
    df = build_velocity_features(df)
    sanity_check_no_future_leakage(df)

    # --- Entity graph features (Day 4) ---
    logger.info("\n--- Entity Graph Features ---")
    df = build_graph_features_fast(df)

    # --- Behavioral features (Day 4) ---
    logger.info("\n--- Behavioral Features ---")
    df = build_behavioral_features(df)

    # --- Time-based features ---
    logger.info("\n--- Time Features ---")
    # Hour of day and day of week (TransactionDT is in seconds from some reference)
    df["hour_of_day"] = (df["TransactionDT"] / 3600 % 24).astype(int)
    df["day_of_week"] = (df["TransactionDT"] / 86400 % 7).astype(int)

    # --- Select modeling columns ---
    # Start with all numeric columns + engineered features
    engineered_cols = [
        # Velocity (Day 3)
        "card1_txn_count_10min", "card1_txn_count_1hr", "card1_txn_count_24hr",
        "device_txn_count_10min", "device_txn_count_1hr", "device_txn_count_24hr",
        "time_since_last_txn_card",
        # Entity graph (Day 4)
        "cards_sharing_device", "cards_sharing_email", "cards_sharing_addr",
        # Behavioral (Day 4)
        "amount_zscore_card", "impossible_travel",
        # Time
        "hour_of_day", "day_of_week",
    ]

    # Original numeric features from the dataset
    original_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and ID
    original_numeric = [
        c for c in original_numeric
        if c not in ["TransactionID", "isFraud"] and c not in engineered_cols
    ]

    all_feature_cols = original_numeric + engineered_cols
    # Only keep columns that actually exist in the dataframe
    all_feature_cols = [c for c in all_feature_cols if c in df.columns]

    modeling_df = df[["TransactionID", "TransactionDT", "isFraud"] + all_feature_cols].copy()

    logger.info(f"\nFinal modeling table shape: {modeling_df.shape}")
    logger.info(f"  Features: {len(all_feature_cols)}")
    logger.info(f"  Fraud rate: {modeling_df['isFraud'].mean():.4%}")

    # Save
    out_path = os.path.join(FEATURES_DIR, "modeling_table.parquet")
    modeling_df.to_parquet(out_path, index=False, chunksize=10000)
    logger.info(f"  Saved to: {out_path}")

    return modeling_df


if __name__ == "__main__":
    build_all_features()


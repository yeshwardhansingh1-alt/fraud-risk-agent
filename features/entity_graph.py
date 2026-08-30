"""
Day 4 — Entity Graph Features.

Cards sharing same device/email/address (fraud ring detection).
Fast groupby-based implementation.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np


def _cumulative_unique(df, group_col, unique_col, out_col):
    if group_col not in df.columns or unique_col not in df.columns:
        df[out_col] = 0
        return df

    # Mark the first time a (group, unique_val) pair appears
    df["_is_new"] = 0
    first_appearances = df[df[group_col].notna()].drop_duplicates(subset=[group_col, unique_col], keep="first")
    df.loc[first_appearances.index, "_is_new"] = 1

    # Cumulative sum of new appearances within each group
    # We must preserve the original index and order
    mask = df[group_col].notna()
    df.loc[mask, out_col] = df[mask].groupby(group_col)["_is_new"].cumsum()
    
    # Fill NAs for rows where group_col is missing
    df[out_col] = df[out_col].fillna(0).astype(int)
    df.drop(columns=["_is_new"], inplace=True)
    return df


def build_graph_features_fast(df):
    """Compute entity sharing counts without future data leakage."""
    logger.info("  Building graph features (leakage-free cumulative method)...")

    df = _cumulative_unique(df, "DeviceInfo", "card1", "cards_sharing_device")
    df = _cumulative_unique(df, "P_emaildomain", "card1", "cards_sharing_email")
    df = _cumulative_unique(df, "addr1", "card1", "cards_sharing_addr")

    return df

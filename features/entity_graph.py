"""
Day 4 — Entity Graph Features.

Cards sharing same device/email/address (fraud ring detection).
Fast groupby-based implementation.
"""

import pandas as pd
import numpy as np


def build_graph_features_fast(df):
    """Compute entity sharing counts using groupby (fast for large datasets)."""
    print("  Building graph features (fast groupby method)...")

    # Cards sharing same device
    if "DeviceInfo" in df.columns:
        device_card_counts = (
            df[df["DeviceInfo"].notna()]
            .groupby("DeviceInfo")["card1"].nunique()
            .rename("cards_sharing_device")
        )
        df = df.merge(device_card_counts, left_on="DeviceInfo", right_index=True, how="left")
        df["cards_sharing_device"] = df["cards_sharing_device"].fillna(0).astype(int)
    else:
        df["cards_sharing_device"] = 0

    # Cards sharing same email domain
    if "P_emaildomain" in df.columns:
        email_card_counts = (
            df[df["P_emaildomain"].notna()]
            .groupby("P_emaildomain")["card1"].nunique()
            .rename("cards_sharing_email")
        )
        df = df.merge(email_card_counts, left_on="P_emaildomain", right_index=True, how="left")
        df["cards_sharing_email"] = df["cards_sharing_email"].fillna(0).astype(int)
    else:
        df["cards_sharing_email"] = 0

    # Cards sharing same address
    if "addr1" in df.columns:
        addr_card_counts = (
            df[df["addr1"].notna()]
            .groupby("addr1")["card1"].nunique()
            .rename("cards_sharing_addr")
        )
        df = df.merge(addr_card_counts, left_on="addr1", right_index=True, how="left")
        df["cards_sharing_addr"] = df["cards_sharing_addr"].fillna(0).astype(int)
    else:
        df["cards_sharing_addr"] = 0

    return df

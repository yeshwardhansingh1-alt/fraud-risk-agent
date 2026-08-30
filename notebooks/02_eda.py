"""
Day 2 — Exploratory Data Analysis on IEEE-CIS Fraud Detection dataset.

Explores TransactionDT, identity columns for entity linking,
fraud rate over time, transaction amount distributions.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def load_merged():
    txn = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"))
    ident = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"))
    return txn.merge(ident, on="TransactionID", how="left")


def explore_transaction_dt(df):
    logger.info("=" * 60)
    logger.info("TransactionDT Exploration")
    logger.info("=" * 60)
    logger.info(f"Min: {df['TransactionDT'].min()}")
    logger.info(f"Max: {df['TransactionDT'].max()}")
    logger.info(f"Range (days): {(df['TransactionDT'].max() - df['TransactionDT'].min()) / 86400:.1f}")
    logger.info(f"Sorted ascending: {df['TransactionDT'].is_monotonic_increasing}")

    df["_day"] = (df["TransactionDT"] - df["TransactionDT"].min()) / 86400
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    df.groupby(df["_day"].astype(int)).size().plot(ax=axes[0], color="#2196F3")
    axes[0].set_title("Transaction Volume per Day")
    axes[0].set_xlabel("Day"); axes[0].set_ylabel("Count")

    daily_fraud = df.groupby(df["_day"].astype(int))["isFraud"].mean()
    daily_fraud.plot(ax=axes[1], color="#F44336")
    axes[1].set_title("Fraud Rate per Day (non-stationarity check)")
    axes[1].axhline(df["isFraud"].mean(), color="gray", linestyle="--", label="Overall mean")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "fraud_rate_over_time.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: plots/fraud_rate_over_time.png")


def identify_identity_columns(df):
    logger.info("\n" + "=" * 60)
    logger.info("Identity Columns for Entity Linking")
    logger.info("=" * 60)
    for col in ["card1","card2","card3","card4","card5","card6",
                "addr1","addr2","P_emaildomain","R_emaildomain","DeviceType","DeviceInfo"]:
        if col in df.columns:
            logger.info(f"  {col:25s}  unique={df[col].nunique():>8,}  null={df[col].isnull().mean():6.2%}")


def plot_amount_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["TransactionAmt"].clip(upper=5000).hist(bins=100, ax=axes[0], color="#4CAF50", alpha=0.7)
    axes[0].set_title("TransactionAmt Distribution (clipped at $5k)")
    axes[0].set_yscale("log")

    fraud = df[df["isFraud"]==1]["TransactionAmt"].clip(upper=2000)
    legit = df[df["isFraud"]==0]["TransactionAmt"].clip(upper=2000)
    axes[1].hist(legit, bins=80, alpha=0.5, color="#2196F3", label="Legit", density=True)
    axes[1].hist(fraud, bins=80, alpha=0.5, color="#F44336", label="Fraud", density=True)
    axes[1].set_title("Amount: Fraud vs. Non-Fraud")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "amount_distribution.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: plots/amount_distribution.png")


if __name__ == "__main__":
    logger.info("Loading data...")
    df = load_merged()
    explore_transaction_dt(df)
    identify_identity_columns(df)
    plot_amount_distribution(df)


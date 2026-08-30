"""
Days 18-19 — Latency Benchmarking.

Send a few thousand requests to the FastAPI endpoint,
record p50/p95/p99 latency, confirm <250ms target is met.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import requests
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

API_URL = "http://localhost:8000"
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def build_sample_requests(n=1000):
    """Build sample requests from the test set."""
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "modeling_table.parquet"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split_info = json.load(f)

    test = df.iloc[split_info["train_size"] + split_info["val_size"]:]
    sample = test.sample(n=min(n, len(test)), random_state=42)

    requests_list = []
    for _, row in sample.iterrows():
        features = {col: float(row[col]) if pd.notna(row[col]) else 0.0
                     for col in feature_cols}
        req = {
            "features": features,
            "transaction_amount": float(row.get("TransactionAmt", 0)),
            "transaction_id": str(int(row["TransactionID"])),
        }
        requests_list.append(req)

    return requests_list


def benchmark(requests_list, n_warmup=10):
    """
    Send requests to the API and measure latency.
    """
    logger.info(f"Benchmarking with {len(requests_list)} requests...")

    # Warmup
    logger.info(f"  Warming up ({n_warmup} requests)...")
    for req in requests_list[:n_warmup]:
        try:
            requests.post(f"{API_URL}/predict", json=req, headers={"X-API-Key": "default-dev-key"}, timeout=5)
        except Exception as e:
            logger.info(f"  Warmup error: {e}")
            return None

    # Benchmark
    latencies = []
    errors = 0

    for i, req in enumerate(requests_list):
        start = time.perf_counter()
        try:
            response = requests.post(f"{API_URL}/predict", json=req, headers={"X-API-Key": "default-dev-key"}, timeout=5)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if response.status_code == 200:
                latencies.append(elapsed_ms)
            else:
                errors += 1
        except Exception:
            errors += 1

        if (i + 1) % 200 == 0:
            logger.info(f"  Sent {i + 1}/{len(requests_list)} requests...")

    latencies = np.array(latencies)

    results = {
        "total_requests": len(requests_list),
        "successful": len(latencies),
        "errors": errors,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(np.mean(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "target_250ms_met": bool(np.percentile(latencies, 99) < 250),
    }

    return results, latencies


def print_results(results):
    """Pretty-print benchmark results."""
    logger.info("\n" + "=" * 60)
    logger.info("LATENCY BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total requests:  {results['total_requests']}")
    logger.info(f"  Successful:      {results['successful']}")
    logger.info(f"  Errors:          {results['errors']}")
    logger.info(f"\n  p50:  {results['p50_ms']:.1f} ms")
    logger.info(f"  p95:  {results['p95_ms']:.1f} ms")
    logger.info(f"  p99:  {results['p99_ms']:.1f} ms")
    logger.info(f"  Mean: {results['mean_ms']:.1f} ms")
    logger.info(f"  Min:  {results['min_ms']:.1f} ms")
    logger.info(f"  Max:  {results['max_ms']:.1f} ms")
    logger.info(f"\n  Target (<250ms p99): {'✓ MET' if results['target_250ms_met'] else '✗ NOT MET'}")
    logger.info("=" * 60)


def plot_latency_histogram(latencies, save_path):
    """Plot latency distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(latencies, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(np.percentile(latencies, 50), color="#4CAF50", linestyle="--",
               label=f"p50 = {np.percentile(latencies, 50):.1f}ms")
    ax.axvline(np.percentile(latencies, 95), color="#FF9800", linestyle="--",
               label=f"p95 = {np.percentile(latencies, 95):.1f}ms")
    ax.axvline(np.percentile(latencies, 99), color="#F44336", linestyle="--",
               label=f"p99 = {np.percentile(latencies, 99):.1f}ms")
    ax.axvline(250, color="black", linestyle=":", label="Target: 250ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Inference Latency Distribution")
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"  Saved latency histogram: {save_path}")


if __name__ == "__main__":
    # Check if API is running
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        logger.info(f"API health: {health.json()}")
    except Exception:
        logger.info("ERROR: API is not running. Start it first with:")
        logger.info("  python agent/api.py")
        exit(1)

    # Build requests
    sample_requests = build_sample_requests(n=2000)

    # Run benchmark
    results, latencies = benchmark(sample_requests)

    if results:
        print_results(results)

        # Plot
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_latency_histogram(latencies, os.path.join(PLOTS_DIR, "latency_histogram.png"))

        # Save results
        with open(os.path.join(MODEL_DIR, "latency_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSaved latency results to: model/latency_results.json")



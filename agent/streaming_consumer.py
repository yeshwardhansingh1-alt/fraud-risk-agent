"""
Mock implementation of a real-time Redis Streams consumer for the Fraud Risk Agent.

In a real production environment, this script runs continuously, fetching 
new transactions from a Redis Stream (e.g., 'txn_stream'), looking up 
pre-computed entity graph features from a fast KV store (Redis), 
scoring the transaction, and publishing the decision to 'decision_stream'.
"""

import os
import json
import logging
import time
import numpy as np
import redis
import joblib
import lightgbm as lgb
from typing import Dict, Any

from agent.expected_loss import argmin_action

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Redis Configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
TXN_STREAM = "txn_stream"
DECISION_STREAM = "decision_stream"
CONSUMER_GROUP = "fraud_scoring_group"
CONSUMER_NAME = "worker_1"

# Global Model Cache
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
calibrated_model = None
feature_cols = None

def load_models():
    """Cache models in memory at startup."""
    global calibrated_model, feature_cols
    logger.info("Loading calibrated model into memory...")
    calibrated_model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "r") as f:
        feature_cols = json.load(f)
    logger.info(f"Loaded {len(feature_cols)} features.")

def process_transaction(r: redis.Redis, txn_id: str, payload: Dict[bytes, bytes]):
    """
    1. Parse incoming raw transaction
    2. Fetch pre-computed stateful features (velocity, graph) from Redis KV
    3. Score the model
    4. Emit decision
    """
    # Parse payload (Assuming JSON serialized in 'data' key)
    try:
        raw_data = json.loads(payload.get(b"data", b"{}").decode('utf-8'))
    except Exception as e:
        logger.error(f"Failed to parse payload for {txn_id}: {e}")
        return

    amount = raw_data.get("TransactionAmt", 0.0)
    
    # In a real setup, we look up stateful features from Redis:
    # e.g. card_velocity = r.get(f"vel:card1:{raw_data['card1']}")
    # For this mock, we assume raw_data already contains all necessary features
    # built by an upstream streaming job (like Flink).
    
    # Build feature vector
    missing = [col for col in feature_cols if col not in raw_data]
    if missing:
        logger.warning(f"Missing {len(missing)} features for {txn_id}. Filling with 0.")
    
    features_2d = np.array([[raw_data.get(col, 0.0) for col in feature_cols]])
    
    # Score
    start_time = time.perf_counter()
    p_fraud = float(calibrated_model.predict_proba(features_2d)[0, 1])
    best_action, losses = argmin_action(p_fraud, amount)
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Construct Decision
    decision = {
        "transaction_id": raw_data.get("TransactionID", txn_id),
        "fraud_probability": round(p_fraud, 6),
        "action": best_action,
        "latency_ms": round(latency_ms, 2)
    }
    
    # Publish to downstream
    r.xadd(DECISION_STREAM, {"decision": json.dumps(decision)})
    logger.info(f"Scored {txn_id} -> {best_action} (p={p_fraud:.4f}) in {latency_ms:.2f}ms")


def run_consumer():
    load_models()
    
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
        r.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except redis.ConnectionError:
        logger.error("Redis is not running. Mock streaming consumer exiting.")
        return

    # Ensure consumer group exists
    try:
        r.xgroup_create(TXN_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
            
    logger.info("Starting Redis Stream consumer...")
    
    while True:
        try:
            # Read from stream, blocking for up to 5 seconds
            messages = r.xreadgroup(CONSUMER_GROUP, CONSUMER_NAME, {TXN_STREAM: ">"}, count=10, block=5000)
            
            for stream_name, events in messages:
                for event_id, payload in events:
                    process_transaction(r, event_id.decode('utf-8'), payload)
                    # Acknowledge successful processing
                    r.xack(TXN_STREAM, CONSUMER_GROUP, event_id)
                    
        except Exception as e:
            logger.error(f"Error reading from stream: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run_consumer()

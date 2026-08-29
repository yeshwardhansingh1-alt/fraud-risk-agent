"""
Days 18-19 — FastAPI endpoint wrapping calibrated model + decision agent.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(__file__) + "/..")
from agent.expected_loss import argmin_action

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

# --- Load model at startup ---
app = FastAPI(
    title="Fraud Risk Agent API",
    description="Cost-sensitive fraud detection with calibrated probabilities and SHAP explainability.",
    version="1.0.0",
)

# Global model objects (loaded once at startup)
calibrated_model = None
booster = None
feature_cols = None


@app.on_event("startup")
async def load_model():
    global calibrated_model, booster, feature_cols

    calibrated_model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm_model.txt"))

    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    print(f"Model loaded with {len(feature_cols)} features.")


class TransactionInput(BaseModel):
    """Input schema for a single transaction."""
    features: Dict[str, float]
    transaction_amount: float = 0.0
    transaction_id: Optional[str] = None


class DecisionOutput(BaseModel):
    """Output schema for a fraud decision."""
    transaction_id: Optional[str]
    fraud_probability: float
    action: str
    expected_losses: Dict[str, float]
    reason: str


@app.post("/predict", response_model=DecisionOutput)
async def predict(txn: TransactionInput):
    """Score a single transaction and return a fraud decision."""
    # Build feature vector
    feature_values = [txn.features.get(col, 0.0) for col in feature_cols]
    features_2d = np.array([feature_values])

    # Get calibrated probability
    p_fraud = float(calibrated_model.predict_proba(features_2d)[0, 1])

    # Argmin action
    best_action, losses = argmin_action(p_fraud, txn.transaction_amount)

    # Build reason
    other_losses = {a: v for a, v in losses.items() if a != best_action}
    next_best = min(other_losses, key=other_losses.get)
    reason = (
        f"{best_action.replace('_', '-').title()} chosen: "
        f"E[loss|{best_action}]=${losses[best_action]:.2f} "
        f"vs E[loss|{next_best}]=${other_losses[next_best]:.2f}"
    )

    return DecisionOutput(
        transaction_id=txn.transaction_id,
        fraud_probability=round(p_fraud, 6),
        action=best_action,
        expected_losses={a: round(v, 2) for a, v in losses.items()},
        reason=reason,
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": calibrated_model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

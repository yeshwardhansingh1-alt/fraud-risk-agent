"""
Days 18-19 — FastAPI endpoint wrapping calibrated model + decision agent.
Now with lifespan events, CORS, SHAP explainer, and a live demo simulator.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Depends, Request
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


from agent.expected_loss import argmin_action
from model.wrappers import CustomCalibratedClassifier, LGBMWrapper

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")
DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "demo")

# Global model objects
calibrated_model = None
booster = None
feature_cols = None
baseline_txns = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model at startup
    global calibrated_model, booster, feature_cols, baseline_txns

    calibrated_model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm_model.txt"))

    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    # Removed SHAP explainer initialization (using LightGBM native)

    # Load baseline transactions (sample 100 from test set)
    logger.info("Loading baseline transactions from test split...")
    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split_info = json.load(f)
        
    train_size = split_info["train_size"]
    val_size = split_info["val_size"]
    test_start = train_size + val_size
    
    # Read a small chunk from the test set for realistic backgrounds
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "modeling_table.parquet"), skiprows=range(1, test_start), nrows=100)
    baseline_txns = df[feature_cols].copy()
    
    logger.info(f"Loaded {len(feature_cols)} features and initialized native explainability.")
    yield
    # Cleanup on shutdown
    pass


app = FastAPI(
    title="Fraud Risk Agent API",
    description="Cost-sensitive fraud detection with calibrated probabilities and SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan
)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API Key Authentication
API_KEY = os.environ.get("FRAUD_API_KEY", "default-dev-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key_header: str = Depends(api_key_header)):
    if not api_key_header or api_key_header != API_KEY:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

# Stricter CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("ALLOWED_ORIGIN", "https://yourdomain.com")],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount static demo files
os.makedirs(DEMO_DIR, exist_ok=True)
app.mount("/demo_ui", StaticFiles(directory=DEMO_DIR, html=True), name="demo")


class TransactionInput(BaseModel):
    """Input schema for a single transaction (raw prediction)."""
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


class DemoInput(BaseModel):
    """Human-readable schema for demo checkout."""
    amount: float
    card_type: str = "visa"
    email_domain: str = "gmail.com"
    device_type: str = "mobile"
    hour: int = 12
    is_repeat_customer: bool = True
    velocity_10min: int = 1
    velocity_1hr: int = 2
    velocity_24hr: int = 5
    cards_on_device: int = 1
    amount_zscore: float = 0.0
    impossible_travel: bool = False


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str


class DemoOutput(BaseModel):
    transaction_id: str
    fraud_probability: float
    action: str
    action_emoji: str
    expected_losses: Dict[str, float]
    reason: str
    top_features: List[ShapFeature]
    risk_level: str


from fastapi import HTTPException

@app.post("/predict", response_model=DecisionOutput, dependencies=[Depends(verify_api_key)])
@limiter.limit("100/minute")
async def predict(request: Request, txn: TransactionInput):
    """Score a single transaction and return a fraud decision."""
    missing_features = [col for col in feature_cols if col not in txn.features]
    if missing_features:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400, 
            detail=f"Missing {len(missing_features)} required features. First 5 missing: {missing_features[:5]}"
        )

    feature_values = [txn.features[col] for col in feature_cols]
    features_2d = np.array([feature_values])

    p_fraud = float(calibrated_model.predict_proba(features_2d)[0, 1])
    best_action, losses = argmin_action(p_fraud, txn.transaction_amount)

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


@app.post("/predict_batch", response_model=List[DecisionOutput], dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def predict_batch(request: Request, txns: List[TransactionInput]):
    """Score a batch of transactions and return fraud decisions."""
    if not txns:
        return []

    # Validate all transactions
    for i, txn in enumerate(txns):
        missing_features = [col for col in feature_cols if col not in txn.features]
        if missing_features:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features in transaction index {i}. First 5 missing: {missing_features[:5]}"
            )

    # Vectorized feature extraction and prediction
    features_2d = np.array([[txn.features[col] for col in feature_cols] for txn in txns])
    p_frauds = calibrated_model.predict_proba(features_2d)[:, 1]

    results = []
    for i, txn in enumerate(txns):
        p_fraud = float(p_frauds[i])
        best_action, losses = argmin_action(p_fraud, txn.transaction_amount)

        other_losses = {a: v for a, v in losses.items() if a != best_action}
        next_best = min(other_losses, key=other_losses.get)
        reason = (
            f"{best_action.replace('_', '-').title()} chosen: "
            f"E[loss|{best_action}]=${losses[best_action]:.2f} "
            f"vs E[loss|{next_best}]=${other_losses[next_best]:.2f}"
        )

        results.append(DecisionOutput(
            transaction_id=txn.transaction_id,
            fraud_probability=round(p_fraud, 6),
            action=best_action,
            expected_losses={a: round(v, 2) for a, v in losses.items()},
            reason=reason,
        ))

    return results


@app.post("/predict_demo", response_model=DemoOutput)
@limiter.limit("20/minute")
async def predict_demo(request: Request, demo: DemoInput):
    """Map human inputs to full feature vector, predict, and explain."""
    # Pick a random baseline transaction to provide realistic V1-V339
    base = baseline_txns.sample(n=1).iloc[0].copy()
    
    # Override with user inputs
    base["TransactionAmt"] = demo.amount
    base["hour_of_day"] = demo.hour
    base["card1_txn_count_10min"] = demo.velocity_10min
    base["card1_txn_count_1hr"] = demo.velocity_1hr
    base["card1_txn_count_24hr"] = demo.velocity_24hr
    base["cards_sharing_device"] = demo.cards_on_device
    base["amount_zscore_card"] = demo.amount_zscore
    base["impossible_travel"] = 1.0 if demo.impossible_travel else 0.0
    
    features_2d = np.array([base.values])
    
    # Predict
    p_fraud = float(calibrated_model.predict_proba(features_2d)[0, 1])
    best_action, losses = argmin_action(p_fraud, demo.amount)
    
    # Emoji
    emoji_map = {"approve": "🟢", "step_up": "🟡", "block": "🔴", "auto_dispute": "🟣"}
    action_emoji = emoji_map.get(best_action, "❓")
    
    # Risk level
    if p_fraud < 0.05: risk = "LOW"
    elif p_fraud < 0.3: risk = "MEDIUM"
    elif p_fraud < 0.8: risk = "HIGH"
    else: risk = "CRITICAL"
    
    # Reason
    other_losses = {a: v for a, v in losses.items() if a != best_action}
    next_best = min(other_losses, key=other_losses.get)
    reason = (
        f"{best_action.replace('_', '-').title()} chosen: "
        f"E[loss|{best_action}]=${losses[best_action]:.2f} "
        f"vs E[loss|{next_best}]=${other_losses[next_best]:.2f}"
    )
    
    # SHAP Explanation via LightGBM native
    shap_values_with_base = booster.predict(features_2d, pred_contrib=True)[0]
    vals = shap_values_with_base[:-1]  # Exclude base value
    
    # Top 3 features by absolute SHAP value
    top_indices = np.argsort(np.abs(vals))[-3:][::-1]
    
    top_features = []
    for idx in top_indices:
        feat_name = feature_cols[idx]
        val = vals[idx]
        direction = "increases fraud risk" if val > 0 else "decreases fraud risk"
        top_features.append(ShapFeature(
            feature=feat_name,
            shap_value=round(float(val), 4),
            direction=direction
        ))
        
    import uuid
    txn_id = f"TXN-{str(uuid.uuid4())[:8].upper()}"

    return DemoOutput(
        transaction_id=txn_id,
        fraud_probability=round(p_fraud, 4),
        action=best_action,
        action_emoji=action_emoji,
        expected_losses={a: round(v, 2) for a, v in losses.items()},
        reason=reason,
        top_features=top_features,
        risk_level=risk
    )


@app.get("/demo/scenarios")
async def get_scenarios():
    """Return pre-built demo scenarios."""
    scenarios_path = os.path.join(DEMO_DIR, "scenarios.json")
    if os.path.exists(scenarios_path):
        with open(scenarios_path) as f:
            return json.load(f)
    return {}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": calibrated_model is not None}

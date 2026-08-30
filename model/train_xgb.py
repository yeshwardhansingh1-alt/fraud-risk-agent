import os
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features")
MODEL_DIR = os.path.dirname(__file__)

def train_xgboost():
    logger.info("Loading modeling table (Parquet)...")
    df = pd.read_parquet(os.path.join(FEATURES_DIR, "modeling_table.parquet"))
    
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "r") as f:
        feature_cols = json.load(f)
        
    with open(os.path.join(MODEL_DIR, "split_info.json"), "r") as f:
        split_info = json.load(f)
        
    train_size = split_info["train_size"]
    val_size = split_info["val_size"]
    
    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size + val_size]
    test = df.iloc[train_size + val_size:]
    
    X_train = train[feature_cols].copy()
    y_train = train["isFraud"].copy()
    X_val = val[feature_cols].copy()
    y_val = val["isFraud"].copy()
    X_test = test[feature_cols].copy()
    y_test = test["isFraud"].copy()

    # Handle NaNs which XGBoost handles natively
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "max_depth": 6,
        "learning_rate": 0.05,
        "nthread": -1,
        "scale_pos_weight": (len(y_train) - sum(y_train)) / sum(y_train)
    }
    
    logger.info("Training XGBoost baseline...")
    evals = [(dtrain, "train"), (dval, "val")]
    bst = xgb.train(params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=30, verbose_eval=False)
    
    # Evaluate
    val_preds = bst.predict(dval)
    test_preds = bst.predict(dtest)
    
    xgb_metrics = {
        "val": {
            "roc_auc": roc_auc_score(y_val, val_preds),
            "pr_auc": average_precision_score(y_val, val_preds)
        },
        "test": {
            "roc_auc": roc_auc_score(y_test, test_preds),
            "pr_auc": average_precision_score(y_test, test_preds)
        }
    }
    
    logger.info("XGBoost Baseline Metrics:")
    logger.info(json.dumps(xgb_metrics, indent=2))
    
    # Save comparison metrics
    with open(os.path.join(MODEL_DIR, "xgb_metrics.json"), "w") as f:
        json.dump(xgb_metrics, f, indent=2)

if __name__ == "__main__":
    train_xgboost()

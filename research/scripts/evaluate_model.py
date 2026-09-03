import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
import json
import joblib

# 1. Load your held-out test set and trained pipeline
# Ensure test_df was NEVER seen during model training/tuning
test_df = pd.read_csv("data/held_out_test_set.csv")
model = joblib.load("models/fraud_risk_xgboost.pkl")

X_test = test_df.drop(columns=["is_fraud", "transaction_id"])
y_true = test_df["is_fraud"].values
amounts = test_df["TransactionAmt"].values  # Required for monetary calculations

# 2. Generate probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# 3. Define Business Parameters (Adjust to match your domain assumptions)
MERCHANT_TAKE_RATE = 0.02   # 2% fee loss on legitimate transaction blocked
CHURN_RATE = 0.05           # 5% of falsely blocked users permanently leave
CUSTOMER_LTV = 1500.0       # ₹1,500 / $1,500 average lifetime value

def calculate_financial_impact(y_true, y_probs, amounts, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Monetary Calculations
    fraud_caught_val = float(np.sum(amounts[(y_pred == 1) & (y_true == 1)]))
    fraud_missed_cost = float(np.sum(amounts[(y_pred == 0) & (y_true == 1)])) # Direct Fraud Loss
    
    fp_amounts = amounts[(y_pred == 1) & (y_true == 0)]
    direct_fp_loss = float(np.sum(fp_amounts * MERCHANT_TAKE_RATE))
    churn_fp_loss = float(fp * CHURN_RATE * CUSTOMER_LTV)
    total_fp_cost = direct_fp_loss + churn_fp_loss
    
    total_business_loss = fraud_missed_cost + total_fp_cost

    return {
        "threshold": round(threshold, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fraud_caught_val": round(fraud_caught_val, 2),
        "fraud_missed_cost": round(fraud_missed_cost, 2),
        "total_fp_cost": round(total_fp_cost, 2),
        "total_business_loss": round(total_business_loss, 2)
    }

# 4. Sweep thresholds to find optimal business operating point
thresholds = np.arange(0.10, 0.95, 0.05)
results = [calculate_financial_impact(y_true, y_probs, amounts, t) for t in thresholds]

# Identify optimal threshold based on lowest business loss
best_run = min(results, key=lambda x: x["total_business_loss"])

# Export metrics artifact for Dashboard & README
evaluation_summary = {
    "total_test_samples": len(y_true),
    "test_fraud_rate": round(float(np.mean(y_true)), 4),
    "optimal_threshold": best_run["threshold"],
    "optimal_metrics": best_run,
    "threshold_sweep": results
}

with open("metrics/evaluation_report.json", "w") as f:
    json.dump(evaluation_summary, f, indent=4)

print("✅ Model evaluation complete. Saved to metrics/evaluation_report.json")

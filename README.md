# Fraud Risk Agent

A cost-sensitive fraud-spike detector built for **Razorpay's AI Buildathon (Track 02: AI Risk Manager)**. The system flags fraudulent card-not-present transactions using a calibrated LightGBM model with honest, auditable metrics — including false-positive cost measured in real dollar terms — evaluated on a strict chronological hold-out set. Every flagged transaction ships with a SHAP-based reason code explaining *why* it was flagged, not just a score. **Purely defensive**: it detects and explains fraud, it does not help commit it.

## Problem

Online fraud costs merchants billions annually through chargebacks, lost revenue, and friction. But blocking too aggressively is equally expensive — false positives frustrate legitimate customers and destroy revenue. Most fraud detectors optimize for accuracy alone, ignoring the asymmetric cost structure. This project builds a system that:

1. **Detects fraud** with a calibrated probability (not just a binary flag)
2. **Minimizes expected loss** across 4 actions: Approve, Step-Up (3DS), Block, Auto-Dispute
3. **Reports honest metrics** including false-positive cost per 1,000 transactions
4. **Explains every decision** with SHAP feature attribution

## Approach

### Data & Features
- **Dataset**: IEEE-CIS Fraud Detection (Kaggle) — 590K+ transactions with real timestamps, device/email/card identity columns
- **Velocity features**: Rolling transaction counts per card/device in 10min, 1hr, 24hr windows
- **Entity graph features**: Cards sharing the same device/email/address (fraud ring detection)
- **Behavioral features**: Amount z-score vs. card's own history, "impossible travel" flag

### Model
- **LightGBM** classifier trained on a **strict chronological split** (no random shuffling)
- **Isotonic calibration** via `CalibratedClassifierCV` — this is the P(Loss|X) engine
- Brier score improvement confirms calibration works

### Decision Agent (Stretch)
- **Expected-loss minimization**: E[Loss(a|X)] = p_fraud × CFraud(a,V) + (1−p_fraud) × CLegit(a,V)
- **Key subtlety**: CFraud(a,V) varies by action — full V+fees under Approve, ~0 under Block, 5% residual leakage under Step-Up. If CFraud is flat, the optimum collapses to "always Approve."
- 4 actions: Approve, Step-Up, Block, Auto-Dispute

### Explainability
- **SHAP TreeExplainer** for feature-level attribution on every flagged transaction
- Top 3-5 contributing features with direction and magnitude
- "Decision receipt" per transaction: `"Step-Up chosen: E[loss|approve]=$46 vs E[loss|step-up]=$3"`

## Architecture

```
fraud-risk-agent/
├── data/                    # IEEE-CIS CSVs (gitignored)
├── features/
│   ├── velocity.py          # Rolling count features (Day 3)
│   ├── entity_graph.py      # Graph-based features (Day 4)
│   ├── behavioral.py        # Z-score, impossible travel (Day 4)
│   └── build_features.py    # Merge all → modeling_table.csv
├── model/
│   ├── rule_engine.py       # Baseline hand-written rules (Day 5)
│   ├── train.py             # LightGBM training (Day 6)
│   ├── calibrate.py         # Isotonic calibration (Day 7)
│   ├── cost_model.py        # Cost assumptions + computation (Day 8)
│   ├── explain.py           # SHAP explainability (Days 12-13)
│   ├── evaluate.py          # Chronological backtest (Day 15)
│   └── net_financial_impact.py  # NFI comparison (Day 16)
├── agent/
│   ├── expected_loss.py     # E[Loss(a|X)] function (Day 9)
│   ├── decision_agent.py    # 4-action policy wrapper (Day 10)
│   ├── dispute_responder.py # Template dispute drafts (Day 11)
│   ├── api.py               # FastAPI endpoint (Days 18-19)
│   └── benchmark.py         # Latency benchmarking
├── dashboard/
│   └── app.py               # Streamlit dashboard (Days 22-25)
├── notebooks/
│   ├── 01_load_and_inspect.py
│   ├── 02_eda.py
│   └── 17_error_analysis.py
├── plots/                   # Generated charts
├── README.md
└── requirements.txt
```

## Key Results

> **Note**: Replace [X], [Y], [Z], [W] with actual numbers after running the pipeline.

| Metric | Value |
|--------|-------|
| Precision | [X]% |
| Recall | [Y]% |
| ROC-AUC | [Z] |
| PR-AUC | [W] |
| FP Cost per 1,000 txn | $[W] |
| Brier Score (calibrated) | [X] |
| p99 Latency | <250ms |

### Cost Assumptions (documented inline)
- Chargeback fee: $25 per chargeback (industry standard $15–$100)
- FP friction cost: $25 per false positive (checkout abandonment)
- Lost revenue fraction: 80% of blocked transaction value
- *Assumption: tune with real data if available*

## How to Run

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Download data (requires Kaggle API credentials)
kaggle competitions download -c ieee-fraud-detection -p data/
# Extract CSVs into data/

# 3. Build features
python features/build_features.py

# 4. Train & calibrate
python model/train.py
python model/calibrate.py

# 5. Evaluate
python model/cost_model.py
python model/evaluate.py

# 6. Run dashboard
streamlit run dashboard/app.py

# 7. (Optional) Run API + benchmark
python agent/api.py &
python agent/benchmark.py
```

## Defense-Only Policy

"Anything offense-capable is disqualified."

This system builds things that flag, score, and respond — never anything that could double as a way to evade detection, probe a live system, or work out how to get a fraudulent transaction past a model. The fraud-spike detector and chargeback evidence responder are both squarely on the safe side of this line.

---

*Built for Razorpay's AI Buildathon (Track 02: AI Risk Manager)*

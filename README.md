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
- **Validation**: Strict schema validation on ingestion via **Pandera**.
- **Storage**: Fully migrated to **Parquet** for ~10x faster I/O and heavily compressed disk footprint.
- **Velocity features**: Rolling transaction counts per card/device in 10min, 1hr, 24hr windows
- **Entity graph features**: Cards sharing the same device/email/address (fraud ring detection)
- **Behavioral features**: Amount z-score vs. card's own history, "impossible travel" flag

### Model
- **LightGBM** classifier trained on a **strict chronological split** (no random shuffling).
- **Hyperparameter Tuning**: Bayesian optimization via **Optuna** built into the training pipeline.
- **Multi-Model Comparison**: Includes an XGBoost baseline script for metric auditing.
- **Isotonic calibration** via `CalibratedClassifierCV` — this is the P(Loss|X) engine
- Brier score improvement confirms calibration works

### Decision Agent (Stretch)
- **Expected-loss minimization**: E[Loss(a|X)] = p_fraud × CFraud(a,V) + (1−p_fraud) × CLegit(a,V)
- **Key subtlety**: CFraud(a,V) varies by action — full V+fees under Approve, ~0 under Block, 5% residual leakage under Step-Up. If CFraud is flat, the optimum collapses to "always Approve."
- 4 actions: Approve, Step-Up, Block, Auto-Dispute
- **Real-Time Streaming**: Includes a mock Redis Stream consumer (`streaming_consumer.py`) for stateful, real-time event scoring.
- **Monitoring**: Built-in Population Stability Index (PSI) hooks to detect feature and prediction distribution drift in production.

### Explainability & API
- **Native LightGBM SHAP feature attribution** (no external heavy C-dependencies needed) on every flagged transaction
- Top 3-5 contributing features with direction and magnitude
- "Decision receipt" per transaction: `"Step-Up chosen: E[loss|approve]=$46 vs E[loss|step-up]=$3"`
- **Secure API**: FastAPI endpoints protected by API Key authentication, `slowapi` rate limiting, and strict CORS. Supports both single `/predict` and bulk `/predict_batch`.

## Architecture

```mermaid
flowchart TD
    %% Data Flow
    subgraph Data Pipeline
        raw_csv[(Kaggle CSVs)]
        features([build_features.py])
        mod_tbl[(modeling_table.csv)]
        
        raw_csv --> features
        features -->|Velocity & Graph| mod_tbl
    end

    %% Training Pipeline
    subgraph Model Training
        train([train.py])
        calib([calibrate.py])
        cost_mod([cost_model.py])
        lgbm[LightGBM Model]
        iso[Isotonic Calibrator]
        
        mod_tbl --> train
        train -->|Hyperopt| lgbm
        mod_tbl --> calib
        calib -->|Uses LGBM| iso
        calib --> cost_mod
    end

    %% Inference Agent
    subgraph Decision Agent
        api([FastAPI Endpoint])
        agent([Decision Logic])
        cost_conf[(cost_config.json)]
        
        iso --> api
        api -->|P_Fraud| agent
        cost_conf --> agent
        agent -->|Approve/Step-Up/Block| Output
    end
    
    Data Pipeline --> Model Training
    Model Training --> Decision Agent
```

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

Based on the chronological hold-out test set (118,108 transactions):

| Metric | Value |
|--------|-------|
| Precision | 75.99% |
| Recall | 33.64% |
| ROC-AUC | 0.8818 |
| PR-AUC | 0.4530 |
| FP Cost per 1,000 txn | $314.10 |
| Missed Fraud Cost per 1,000 txn | $4,772.71 |
| Net Savings (over approve-all) | $369,395.18 (Agent Logic) |
| NFI Improvement (Agent vs Rules) | $6,614,663.36 (out of ~$15.8M total test volume) |
| p99 Latency (Inference) | < 15ms |

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
pip install -e .

# 2. Download data (requires Kaggle API credentials)
kaggle competitions download -c ieee-fraud-detection -p data/
# Extract CSVs into data/

# 3. Run Pipeline (using Makefile)
make all

# Note: 'make all' automates:
# - build_features.py (Pandera schema checks -> Parquet)
# - train.py (Optuna tuning -> LightGBM)
# - calibrate.py (Isotonic Regression)
# - rule_engine.py & evaluate.py & net_financial_impact.py

# 4. Run the Live Demo Simulator (FastAPI)
make serve
# Or: uvicorn agent.api:app --host 127.0.0.1 --port 8000

# 5. Visit http://localhost:8000/demo_ui in your browser to test live checkout transactions!
```

## Defense-Only Policy

"Anything offense-capable is disqualified."

This system builds things that flag, score, and respond — never anything that could double as a way to evade detection, probe a live system, or work out how to get a fraudulent transaction past a model. The fraud-spike detector and chargeback evidence responder are both squarely on the safe side of this line.

---

*Built for Razorpay's AI Buildathon (Track 02: AI Risk Manager)*

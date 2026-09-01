# Real-Time Fraud Risk Agent (Razorpay Buildathon Track 02)

An enterprise-grade, event-driven AI Risk Manager designed specifically for Payment Aggregator (PA/PG) economics. Built to address Track 02 (AI Risk Manager) of the Razorpay AI Buildathon, this system transitions beyond raw statistical metrics (AUC/F1) by evaluating live transaction streams against a custom Net Financial Impact (NFI) optimizer.

Every flagged transaction is paired with real-time TreeSHAP feature attributions, guarded by an active Circuit Breaker failsafe, and logged to an immutable audit trail with an automated Chargeback Evidence Generator for merchant representment.

## 1. Track 02 Requirements Mapping

| Track 02 Requirement | System Implementation | Verification Artifact |
| --- | --- | --- |
| **Honest Metrics & Cost Model** | Custom Net Financial Impact (NFI) utility metric calibrating Gateway Revenue Loss ($C_{friction}$) against Chargeback Liabilities ($C_{chargeback}$). | `app/ml/cost_optimizer.py` |
| **Explainable & Bounded** | Real-time TreeSHAP local feature attribution returning top reason codes. Tiered actions bounded by calibrated probability cutoffs (PASS, STEP_UP, BLOCK). | `app/ml/inference.py` & `app/agent/policy.py` |
| **Failure Handled Gracefully** | 5-minute rolling window Circuit Breaker monitoring block rate. Trips to ACTION_STEP_UP if anomaly rate exceeds 8%. | `app/agent/circuit_breaker.py` |
| **Working Auto-Responder** | Chargeback Evidence Auto-Assembler compiling SHAP vectors, velocity metrics, and device fingerprints into representment JSON payloads. | `app/agent/auto_responder.py` |
| **Strictly Defense-Only** | Exclusively reactive detection, risk scoring, failsafe policy execution, and dispute resolution artifacts. | Complete Codebase |

## 2. System Architecture

The pipeline replaces heavy synchronous batch operations with a zero-bloat, decoupled streaming pipeline running via Redis Streams and FastAPI, backed by a low-overhead SQLite Write-Ahead Logging (WAL) audit ledger.

```plaintext
[ Incoming Transaction Stream / Mock Generator ]
│
▼
[ FastAPI Ingestion Endpoint ] /v1/risk/evaluate
│ (Async Pipeline)
▼
[ Redis Feature Store ]
├── Stream Queue: 'tx_stream'
└── TTL Hashes: 'card:{id}:count_15m'
│
▼
[ Hydration & ML Scoring ]
├── Feature Hydrator (Microsecond Lookup)
├── LightGBM Inference (< 10ms)
└── TreeSHAP Reason Code Generator
│
▼
[ Bounded Policy & Failsafe ]
├── Action Engine (Pass / Step-Up / Block)
└── Circuit Breaker (8% Anomaly Guard)
│
▼
[ Immutable SQLite Audit Ledger ]
│
▼
[ Streamlit Real-Time Dashboard ]
```

## 3. Performance & Benchmarks

By utilizing atomic Redis pipelines for sliding-window velocity aggregations and in-memory LightGBM scoring, the event loop remains unblocked under high concurrency.

* **Concurrency Target:** 100 concurrent workers
* **p95 Latency:** < 30 ms
* **p99 Latency:** < 45 ms
* **Memory Footprint:** < 100 MB (Optimized for 8GB RAM local deployment)

### Load Test Execution

Run the included Locust suite to verify latency claims locally:

```bash
locust -f locustfile.py --headless -u 100 -r 10 --run-time 1m --host http://localhost:8000
```

## 4. Key Results (Chronological Hold-Out Set)

Evaluated across 118,108 transactions using strict chronological splitting to eliminate target leakage:

| Metric / Dimension | Value / Economic Impact |
| --- | --- |
| **ROC-AUC / PR-AUC** | 0.8818 / 0.4530 |
| **Precision / Recall** | 75.99% / 33.64% |
| **False Positive Cost (per 1k txns)** | $314.10 (Lost Merchant Margin) |
| **Missed Fraud Cost (per 1k txns)** | $4,772.71 (Chargeback Penalties + Loss) |
| **Net Financial Savings (vs. Approve All)**| +$369,395.18 |
| **NFI Improvement (Agent vs. Base Rules)** | +$6,614,663.36 |

## 5. Repository Structure

```plaintext
fraud-risk-agent/
├── app/
│   ├── main.py                # Async FastAPI Gateway & Background Logging
│   ├── config.py              # System Configuration & Thresholds
│   ├── streaming/
│   │   ├── redis_client.py    # Atomic Redis Feature Hydrator & Pipelines
│   │   └── consumer.py        # Event Stream Worker
│   ├── ml/
│   │   ├── inference.py       # LightGBM Inference & Real-Time TreeSHAP
│   │   └── cost_optimizer.py  # Business Net Financial Impact Calculator
│   ├── agent/
│   │   ├── policy.py          # Tiered Bounded Policy Engine
│   │   ├── circuit_breaker.py # 8% Anomaly Failsafe Guard
│   │   └── auto_responder.py  # Chargeback Evidence Representment Assembler
│   └── database/
│       └── ledger.py          # SQLite WAL-mode Immutable Audit DB
├── dashboard.py               # Real-Time Streamlit Visual UI
├── docker-compose.yml         # Docker Orchestration Configuration
├── Dockerfile                 # Containerization Spec
├── locustfile.py              # Load Benchmark Suite
└── requirements.txt           # Production Dependencies
```

## 6. Quickstart Guide

### Prerequisites
* Python 3.10+
* Docker Desktop (for Redis & Containerized Setup)

### Option A: Running via Docker Compose (Recommended)

Clone the repository:
```bash
git clone https://github.com/yeshwardhansingh1-alt/fraud-risk-agent.git
cd fraud-risk-agent
```

Spin up Redis and the FastAPI Backend:
```bash
docker-compose up -d
```

Launch the Streamlit Live Dashboard:
```bash
streamlit run dashboard.py
```

### Option B: Local Python Development Setup

Initialize Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Start Redis Container (256MB Cap):
```bash
docker run -d --name fraud-redis -p 6379:6379 --memory="256m" redis:alpine
```

Launch the API Gateway:
```bash
uvicorn app.main:app --reload --port 8000
```

Launch the Dashboard:
```bash
streamlit run dashboard.py
```

## 7. Defense-Only Statement

This project is strictly defense-only. All modules are designed to detect, score, gate, and defend against financial loss. It contains zero adversarial payload generators, attack simulators, or evasion scripts. All automated outputs are strictly for audit trails, real-time risk mitigation, and merchant chargeback defense.

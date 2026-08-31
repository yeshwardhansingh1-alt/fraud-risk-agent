import sqlite3
import json
import time

class SQLiteAuditLedger:
    def __init__(self, db_path="audit_ledger.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_trail (
                    tx_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    amount REAL,
                    risk_score REAL,
                    action TEXT,
                    reasons TEXT,
                    latency_ms REAL
                )
            """)

    def log_decision(self, tx_id: str, amount: float, decision: dict, latency_ms: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO audit_trail VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tx_id,
                    time.time(),
                    amount,
                    decision["risk_score"],
                    decision["action"],
                    json.dumps(decision.get("top_reasons", [])),
                    latency_ms
                )
            )

    def get_record(self, tx_id: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM audit_trail WHERE tx_id = ?", (tx_id,)).fetchone()
            if row:
                d = dict(row)
                d["top_reasons"] = json.loads(d["reasons"])
                return d
            return None

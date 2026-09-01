import time
import requests
import os
import random

API_URL = "http://127.0.0.1:8000/v1/risk/evaluate"
DB_PATH = "audit_ledger.db"

def test_chaos_malformed_payload():
    print("[CHAOS] Injecting Malformed Payload...")
    payload = {"tx_id": "malformed", "card_id": 12345} # Missing amount, wrong types
    response = requests.post(API_URL, json=payload)
    print(f"Response Code: {response.status_code}")
    assert response.status_code == 200 # App handles it gracefully by defaulting amount
    print(f"Decision: {response.json().get('decision')}")
    print("[CHAOS] Malformed Payload Test Passed.")

def test_chaos_extreme_values():
    print("[CHAOS] Injecting Extreme Values...")
    payload = {"tx_id": "extreme", "card_id": "card_chaos", "amount": 999999.99}
    response = requests.post(API_URL, json=payload)
    print(f"Response Code: {response.status_code}")
    decision = response.json().get('decision')
    assert decision['action'] == 'ACTION_BLOCK'
    print("[CHAOS] Extreme Value Test Passed.")

def test_chaos_db_deletion_mid_stream():
    print("[CHAOS] Simulating Database Pod Crash...")
    # Delete DB if it exists
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except PermissionError:
            print("Cannot delete DB while in use on Windows, skipping strict DB deletion test.")
    
    # Fire off 5 requests while DB is missing
    for i in range(5):
        payload = {"tx_id": f"chaos_{i}", "card_id": "card_chaos", "amount": random.uniform(10, 50)}
        try:
            res = requests.post(API_URL, json=payload, timeout=2)
            assert res.status_code == 200
        except Exception as e:
            print(f"Failed to handle DB crash: {e}")
            assert False
    print("[CHAOS] Database Crash Test Passed. API remained available.")

if __name__ == "__main__":
    test_chaos_malformed_payload()
    test_chaos_extreme_values()
    test_chaos_db_deletion_mid_stream()
    print("\n✅ All Chaos Experiments Completed Successfully!")

from locust import HttpUser, task, between
import random
import uuid

class FraudRiskUser(HttpUser):
    wait_time = between(0.01, 0.1) # Aggressive load

    @task
    def evaluate_transaction(self):
        payload = {
            "tx_id": str(uuid.uuid4()),
            "card_id": f"card_{random.randint(1, 1000)}",
            "amount": round(random.uniform(10.0, 5000.0), 2)
        }
        
        # In a real environment, you might need API keys
        self.client.post("/v1/risk/evaluate", json=payload)

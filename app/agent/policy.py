from app.agent.circuit_breaker import CircuitBreaker

class ActionPolicyEngine:
    def __init__(self, opt_threshold: float):
        self.opt_threshold = opt_threshold
        self.circuit_breaker = CircuitBreaker()

    def evaluate(self, prob: float, amount: float, shap_reasons: list) -> dict:
        # Check Circuit Breaker State First
        if self.circuit_breaker.record_and_check("CHECK"):
            return {
                "action": "ACTION_STEP_UP",
                "reason": "CIRCUIT_BREAKER_TRIPPED_SYSTEM_FAILSAFE",
                "risk_score": prob
            }
            
        # Tiered Bounded Policy Execution
        if prob < (self.opt_threshold * 0.7):
            action = "ACTION_PASS"
        elif prob < self.opt_threshold:
            action = "ACTION_STEP_UP"  # 3DS / OTP Challenge
        else:
            action = "ACTION_BLOCK"
            
        self.circuit_breaker.record_and_check(action)
        
        return {
            "action": action,
            "risk_score": prob,
            "threshold_used": self.opt_threshold,
            "top_reasons": shap_reasons
        }

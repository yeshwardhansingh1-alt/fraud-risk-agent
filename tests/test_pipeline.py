import pytest
from app.agent.circuit_breaker import CircuitBreaker
from app.agent.policy import ActionPolicyEngine
import time

def test_circuit_breaker_trips():
    breaker = CircuitBreaker(window_seconds=300, max_block_rate=0.08)
    
    # Needs at least 20 samples to trip
    for _ in range(18):
        breaker.record_and_check("ACTION_PASS")
    
    # 18 pass, 2 blocks -> 2 / 20 = 10%, which is > 8%
    breaker.record_and_check("ACTION_BLOCK")
    is_tripped = breaker.record_and_check("ACTION_BLOCK")
    
    assert is_tripped == True

def test_circuit_breaker_does_not_trip():
    breaker = CircuitBreaker(window_seconds=300, max_block_rate=0.08)
    
    # 20 samples, all pass -> 0% block rate
    for _ in range(20):
        is_tripped = breaker.record_and_check("ACTION_PASS")
        assert is_tripped == False

def test_policy_engine_circuit_breaker_integration():
    engine = ActionPolicyEngine(opt_threshold=0.5)
    
    # Push circuit breaker past limit
    for _ in range(18):
        engine.circuit_breaker.record_and_check("ACTION_PASS")
    for _ in range(3):
        engine.circuit_breaker.record_and_check("ACTION_BLOCK")
        
    # The next evaluation should fail safe
    decision = engine.evaluate(0.9, 100.0, [])
    assert decision["action"] == "ACTION_STEP_UP"
    assert decision["reason"] == "CIRCUIT_BREAKER_TRIPPED_SYSTEM_FAILSAFE"

import pytest
from agent.expected_loss import cfraud, clegit, expected_loss, argmin_action

def test_cfraud():
    # Test Approve
    assert cfraud("approve", 100.0) == 125.0
    # Test Block
    assert cfraud("block", 100.0) == 0.0

def test_clegit():
    # Test Approve
    assert clegit("approve", 100.0) == 0.0
    # Test Block (Friction + 0.8 * V)
    assert clegit("block", 100.0) == 25.0 + 80.0

def test_argmin_action():
    V = 100.0
    
    # Low p_fraud -> approve
    best, losses = argmin_action(0.001, V)
    assert best == "approve"
    
    # Moderate p_fraud -> step_up
    best, losses = argmin_action(0.10, V)
    assert best == "step_up"
    
    # High p_fraud -> block
    best, losses = argmin_action(0.80, V)
    assert best == "block"

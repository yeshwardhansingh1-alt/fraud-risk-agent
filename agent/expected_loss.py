"""
Day 9 — Expected Loss Function [STRETCH].

E[Loss(a|X)] for all four actions: Approve, Step-Up, Block, Auto-Dispute.
CFraud(a, V) definition.

Subtlety: CFraud can't be a flat function of V alone, or the fraud term
is constant across all four actions and the optimum collapses to "always Approve."
Make it CFraud(a, V) — full V+fees under Approve, ~0 under Block,
a small residual leakage rate under Step-Up since some fraud still slips past 3DS.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import numpy as np


from model.cost_model import COST_CONFIG

# --- Cost of Fraud given action and transaction value ---
# CFraud(a, V): cost if the transaction IS fraud AND we take action a.

from typing import Tuple, Dict

def cfraud(action: str, V: float) -> float:
    """
    Cost of fraud given action and transaction value.
    """
    CHARGEBACK_FEE = COST_CONFIG["chargeback_fee"]
    DISPUTE_PROCESSING_FEE = COST_CONFIG.get("dispute_processing_fee", 10.0)
    STEP_UP_LEAKAGE_RATE = COST_CONFIG.get("step_up_leakage_rate", 0.25)
    BLOCK_COST = 0.0

    costs = {
        "approve": V * COST_CONFIG["fraud_loss_fraction"] + CHARGEBACK_FEE,
        "step_up": STEP_UP_LEAKAGE_RATE * (V * COST_CONFIG["fraud_loss_fraction"] + CHARGEBACK_FEE),
        "block": BLOCK_COST,
        "auto_dispute": DISPUTE_PROCESSING_FEE,
    }
    return costs.get(action, 0.0)


# --- Cost of False Positive given action and transaction value ---
# CLegit(a, V): cost if the transaction is LEGITIMATE and we take action a.

def clegit(action: str, V: float) -> float:
    """
    Cost when the transaction is legitimate and we take action a.
    """
    STEP_UP_FRICTION = COST_CONFIG.get("step_up_friction_cost", 5.0)
    BLOCK_FRICTION = COST_CONFIG["false_positive_friction_cost"]
    BLOCK_LOST_REVENUE_FRACTION = COST_CONFIG["lost_revenue_fraction"]

    costs = {
        "approve": 0.0,
        "step_up": STEP_UP_FRICTION,
        "block": BLOCK_FRICTION + BLOCK_LOST_REVENUE_FRACTION * V,
        "auto_dispute": BLOCK_FRICTION + BLOCK_LOST_REVENUE_FRACTION * V,  # Massive friction if we dispute legit txn
    }
    return costs.get(action, 0.0)


# --- Expected Loss ---

ACTIONS = ["approve", "step_up", "block", "auto_dispute"]


def expected_loss(action: str, p_fraud: float, V: float) -> float:
    """
    E[Loss(a|X)] = p_fraud * CFraud(a, V) + (1 - p_fraud) * CLegit(a, V)

    Args:
        action: one of "approve", "step_up", "block", "auto_dispute"
        p_fraud: calibrated probability of fraud P(Fraud|X)
        V: transaction value (TransactionAmt)

    Returns:
        Expected loss for taking this action
    """
    return p_fraud * cfraud(action, V) + (1 - p_fraud) * clegit(action, V)


def argmin_action(p_fraud: float, V: float) -> Tuple[str, Dict[str, float]]:
    """
    Find the action that minimizes expected loss.

    Returns (best_action, expected_losses_dict)
    """
    losses = {a: expected_loss(a, p_fraud, V) for a in ACTIONS}
    best_action = min(losses, key=losses.get)
    return best_action, losses


# ============================================================
# Unit Tests
# ============================================================

def test_expected_loss():
    """
    Unit test: confirm that as fraud probability rises,
    the argmin shifts from Approve → Step-Up → Block as expected.
    """
    V = 100.0  # Test with a $100 transaction

    logger.info("Unit test: E[Loss(a|X)] argmin behavior")
    logger.info(f"  Transaction value: ${V}")
    logger.info()

    # At very low p_fraud, Approve should win
    best, losses = argmin_action(0.001, V)
    logger.info(f"  p_fraud=0.001:  best={best:12s}  losses={_fmt_losses(losses)}")
    assert best == "approve", f"Expected 'approve' at p=0.001, got '{best}'"

    # At moderate p_fraud, Step-Up should be optimal
    best, losses = argmin_action(0.10, V)
    logger.info(f"  p_fraud=0.10:   best={best:12s}  losses={_fmt_losses(losses)}")
    # Step-Up should beat Approve here
    assert losses["step_up"] < losses["approve"], "Step-Up should beat Approve at p=0.10"

    # At high p_fraud, Block should win
    best, losses = argmin_action(0.80, V)
    logger.info(f"  p_fraud=0.80:   best={best:12s}  losses={_fmt_losses(losses)}")
    assert best == "block", f"Expected 'block' at p=0.80, got '{best}'"

    # At p_fraud=1.0, Block should clearly win (zero cost vs full loss)
    best, losses = argmin_action(1.0, V)
    logger.info(f"  p_fraud=1.00:   best={best:12s}  losses={_fmt_losses(losses)}")
    assert best == "block", f"Expected 'block' at p=1.0, got '{best}'"

    # Verify that CFraud(approve, V) > CFraud(step_up, V) > CFraud(block, V)
    assert cfraud("approve", V) > cfraud("step_up", V) > cfraud("block", V), \
        "CFraud ordering should be: approve > step_up > block"

    # Verify that CLegit(block, V) > CLegit(step_up, V) > CLegit(approve, V)
    assert clegit("block", V) > clegit("step_up", V) > clegit("approve", V), \
        "CLegit ordering should be: block > step_up > approve"

    logger.info("\n  [OK] All unit tests passed!")
    return True


def _fmt_losses(losses):
    return "  ".join(f"{a}=${v:.2f}" for a, v in losses.items())


if __name__ == "__main__":
    test_expected_loss()

    # Show decision boundary sweep
    logger.info("\n\nDecision boundary sweep (V=$200):")
    logger.info(f"{'p_fraud':>10s}  {'best_action':>15s}  {'E[approve]':>12s}  {'E[step_up]':>12s}  {'E[block]':>12s}")
    logger.info("-" * 70)
    for p in np.arange(0, 1.01, 0.05):
        best, losses = argmin_action(p, 200.0)
        logger.info(f"{p:10.2f}  {best:>15s}  ${losses['approve']:>10.2f}  ${losses['step_up']:>10.2f}  ${losses['block']:>10.2f}")



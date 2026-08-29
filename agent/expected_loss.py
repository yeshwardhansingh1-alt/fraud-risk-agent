"""
Day 9 — Expected Loss Function [STRETCH].

E[Loss(a|X)] for all four actions: Approve, Step-Up, Block, Auto-Dispute.
CFraud(a, V) definition.

Subtlety: CFraud can't be a flat function of V alone, or the fraud term
is constant across all four actions and the optimum collapses to "always Approve."
Make it CFraud(a, V) — full V+fees under Approve, ~0 under Block,
a small residual leakage rate under Step-Up since some fraud still slips past 3DS.
"""

import numpy as np


# --- Cost of Fraud given action and transaction value ---
# CFraud(a, V): cost if the transaction IS fraud AND we take action a.

def cfraud(action, V):
    """
    Cost of fraud given action and transaction value.

    - Approve: full loss = V + chargeback fee ($25)
    - Step-Up (3DS): residual leakage rate (~5% still slip through)
      so cost = 0.05 * (V + $25)
    - Block: ~$0 (blocked, no loss — maybe tiny investigation cost)
    - Auto-Dispute: cost = $10 dispute processing fee (fraud already happened,
      we're just disputing it, recover most of V)
    """
    CHARGEBACK_FEE = 25.0
    DISPUTE_PROCESSING_FEE = 10.0
    STEP_UP_LEAKAGE_RATE = 0.25  # 25% of fraud still slips through 3DS
    BLOCK_COST = 0.0

    costs = {
        "approve": V + CHARGEBACK_FEE,
        "step_up": STEP_UP_LEAKAGE_RATE * (V + CHARGEBACK_FEE),
        "block": BLOCK_COST,
        "auto_dispute": DISPUTE_PROCESSING_FEE,
    }
    return costs.get(action, 0)


# --- Cost of False Positive given action and transaction value ---
# CLegit(a, V): cost if the transaction is LEGITIMATE and we take action a.

def clegit(action, V):
    """
    Cost when the transaction is legitimate and we take action a.

    - Approve: $0 (correct decision)
    - Step-Up: friction cost ~$5 (customer has to go through 3DS, some abandon)
    - Block: lost revenue = fraction of V + friction
    - Auto-Dispute: $0 (only triggered on flagged first-party fraud, not here)
    """
    STEP_UP_FRICTION = 5.0
    BLOCK_FRICTION = 25.0
    BLOCK_LOST_REVENUE_FRACTION = 0.80

    costs = {
        "approve": 0.0,
        "step_up": STEP_UP_FRICTION,
        "block": BLOCK_FRICTION + BLOCK_LOST_REVENUE_FRACTION * V,
        "auto_dispute": BLOCK_FRICTION + BLOCK_LOST_REVENUE_FRACTION * V,  # Massive friction if we dispute legit txn
    }
    return costs.get(action, 0)


# --- Expected Loss ---

ACTIONS = ["approve", "step_up", "block", "auto_dispute"]


def expected_loss(action, p_fraud, V):
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


def argmin_action(p_fraud, V):
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

    print("Unit test: E[Loss(a|X)] argmin behavior")
    print(f"  Transaction value: ${V}")
    print()

    # At very low p_fraud, Approve should win
    best, losses = argmin_action(0.001, V)
    print(f"  p_fraud=0.001:  best={best:12s}  losses={_fmt_losses(losses)}")
    assert best == "approve", f"Expected 'approve' at p=0.001, got '{best}'"

    # At moderate p_fraud, Step-Up should be optimal
    best, losses = argmin_action(0.10, V)
    print(f"  p_fraud=0.10:   best={best:12s}  losses={_fmt_losses(losses)}")
    # Step-Up should beat Approve here
    assert losses["step_up"] < losses["approve"], "Step-Up should beat Approve at p=0.10"

    # At high p_fraud, Block should win
    best, losses = argmin_action(0.80, V)
    print(f"  p_fraud=0.80:   best={best:12s}  losses={_fmt_losses(losses)}")
    assert best == "block", f"Expected 'block' at p=0.80, got '{best}'"

    # At p_fraud=1.0, Block should clearly win (zero cost vs full loss)
    best, losses = argmin_action(1.0, V)
    print(f"  p_fraud=1.00:   best={best:12s}  losses={_fmt_losses(losses)}")
    assert best == "block", f"Expected 'block' at p=1.0, got '{best}'"

    # Verify that CFraud(approve, V) > CFraud(step_up, V) > CFraud(block, V)
    assert cfraud("approve", V) > cfraud("step_up", V) > cfraud("block", V), \
        "CFraud ordering should be: approve > step_up > block"

    # Verify that CLegit(block, V) > CLegit(step_up, V) > CLegit(approve, V)
    assert clegit("block", V) > clegit("step_up", V) > clegit("approve", V), \
        "CLegit ordering should be: block > step_up > approve"

    print("\n  [OK] All unit tests passed!")
    return True


def _fmt_losses(losses):
    return "  ".join(f"{a}=${v:.2f}" for a, v in losses.items())


if __name__ == "__main__":
    test_expected_loss()

    # Show decision boundary sweep
    print("\n\nDecision boundary sweep (V=$200):")
    print(f"{'p_fraud':>10s}  {'best_action':>15s}  {'E[approve]':>12s}  {'E[step_up]':>12s}  {'E[block]':>12s}")
    print("-" * 70)
    for p in np.arange(0, 1.01, 0.05):
        best, losses = argmin_action(p, 200.0)
        print(f"{p:10.2f}  {best:>15s}  ${losses['approve']:>10.2f}  ${losses['step_up']:>10.2f}  ${losses['block']:>10.2f}")

    print("\nDay 9 complete.")

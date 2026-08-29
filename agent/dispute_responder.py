"""
Day 11 — Auto-Dispute Responder [STRETCH].

For flagged first-party/friendly-fraud cases, build a template-based dispute draft.
Template cites evidence fields: AVS match, CVV match, 3DS liability shift,
device match, prior good history.

Purely defensive — drafting evidence for a dispute you're already facing,
not evading anything.
"""

import json
from datetime import datetime


# --- Evidence field extractors ---

def get_avs_match(transaction):
    """Check Address Verification Service match."""
    # In IEEE-CIS data, we use card address fields as proxy
    addr1 = transaction.get("addr1")
    addr2 = transaction.get("addr2")
    if addr1 is not None and addr2 is not None:
        return "AVS_MATCH"
    elif addr1 is not None:
        return "AVS_PARTIAL"
    return "AVS_UNAVAILABLE"


def get_cvv_match(transaction):
    """Check CVV verification. In IEEE-CIS, card4/card6 as proxy."""
    card4 = transaction.get("card4")  # card network
    card6 = transaction.get("card6")  # card type (debit/credit)
    if card4 is not None and card6 is not None:
        return "CVV_MATCH"
    return "CVV_UNAVAILABLE"


def get_3ds_status(transaction):
    """Check 3D Secure status. No direct field in IEEE-CIS, use ProductCD proxy."""
    product = transaction.get("ProductCD")
    if product in ("W", "H"):
        return "3DS_ENROLLED"
    return "3DS_NOT_ENROLLED"


def get_device_match(transaction):
    """Check if device matches known customer devices."""
    device_info = transaction.get("DeviceInfo")
    device_type = transaction.get("DeviceType")
    if device_info is not None:
        return f"DEVICE_IDENTIFIED: {device_type} / {device_info}"
    return "DEVICE_UNKNOWN"


def get_prior_history(transaction, velocity_features=None):
    """Check cardholder's prior good transaction history."""
    card_count_24hr = transaction.get("card1_txn_count_24hr", 0)
    if card_count_24hr > 5:
        return f"GOOD_HISTORY: {card_count_24hr} transactions in last 24hr"
    elif card_count_24hr > 0:
        return f"LIMITED_HISTORY: {card_count_24hr} transactions in last 24hr"
    return "NO_PRIOR_HISTORY"


# --- Dispute Template ---

DISPUTE_TEMPLATE = """
================================================================================
CHARGEBACK DISPUTE RESPONSE — DRAFT
================================================================================
Date:              {date}
Transaction ID:    {transaction_id}
Transaction Amount: ${amount:.2f}
Card Type:         {card_type}

--- FRAUD PROBABILITY ASSESSMENT ---
Model Confidence:  {fraud_probability:.2%} probability of fraud
Agent Decision:    {action}
Reason:            {reason}

--- EVIDENCE SUMMARY ---

1. ADDRESS VERIFICATION (AVS):
   Status: {avs_status}

2. CVV VERIFICATION:
   Status: {cvv_status}

3. 3D SECURE (3DS):
   Status: {three_ds_status}
   {liability_note}

4. DEVICE IDENTIFICATION:
   Status: {device_status}

5. CARDHOLDER HISTORY:
   Status: {prior_history}

--- SUPPORTING EVIDENCE ---
{supporting_evidence}

--- RECOMMENDATION ---
{recommendation}

================================================================================
This is a template-based draft. Review and customize before submission.
Defense-only: this document supports dispute resolution, not fraud evasion.
================================================================================
"""


def generate_dispute_response(transaction, decision=None):
    """
    Generate a template-based dispute response for a flagged transaction.

    Args:
        transaction: dict with transaction fields
        decision: optional decision dict from the decision agent

    Returns:
        Formatted dispute response string
    """
    # Extract evidence
    avs = get_avs_match(transaction)
    cvv = get_cvv_match(transaction)
    three_ds = get_3ds_status(transaction)
    device = get_device_match(transaction)
    history = get_prior_history(transaction)

    # 3DS liability note
    if three_ds == "3DS_ENROLLED":
        liability_note = "Liability shift applies: issuer bears fraud liability."
    else:
        liability_note = "No liability shift: merchant bears fraud liability."

    # Build supporting evidence list
    evidence_items = []
    if avs in ("AVS_MATCH", "AVS_PARTIAL"):
        evidence_items.append("- Billing address verified via AVS")
    if cvv == "CVV_MATCH":
        evidence_items.append("- CVV code verified at transaction time")
    if three_ds == "3DS_ENROLLED":
        evidence_items.append("- 3D Secure authentication completed")
    if "DEVICE_IDENTIFIED" in device:
        evidence_items.append(f"- Transaction device identified: {device.split(': ')[1]}")
    if "GOOD_HISTORY" in history:
        evidence_items.append(f"- Cardholder has positive transaction history: {history.split(': ')[1]}")

    if not evidence_items:
        evidence_items.append("- Limited evidence available for this transaction")

    supporting_evidence = "\n".join(evidence_items)

    # Recommendation
    fraud_prob = decision.get("fraud_probability", 0) if decision else 0
    if fraud_prob > 0.8:
        recommendation = (
            "HIGH FRAUD PROBABILITY. Strong evidence of fraudulent activity. "
            "Recommend accepting the chargeback unless compelling counter-evidence exists."
        )
    elif fraud_prob > 0.5:
        recommendation = (
            "MODERATE FRAUD PROBABILITY. Mixed signals. "
            "Review the evidence above and consider reaching out to the cardholder "
            "before deciding whether to contest."
        )
    else:
        recommendation = (
            "LOW FRAUD PROBABILITY. Likely first-party/friendly fraud. "
            "Contest the chargeback with the evidence documented above."
        )

    # Format template
    response = DISPUTE_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        transaction_id=transaction.get("TransactionID", "N/A"),
        amount=transaction.get("TransactionAmt", 0),
        card_type=f"{transaction.get('card4', 'N/A')} / {transaction.get('card6', 'N/A')}",
        fraud_probability=fraud_prob,
        action=decision.get("action", "N/A") if decision else "N/A",
        reason=decision.get("reason", "N/A") if decision else "N/A",
        avs_status=avs,
        cvv_status=cvv,
        three_ds_status=three_ds,
        liability_note=liability_note,
        device_status=device,
        prior_history=history,
        supporting_evidence=supporting_evidence,
        recommendation=recommendation,
    )

    return response


if __name__ == "__main__":
    # Demo with a sample transaction
    sample_txn = {
        "TransactionID": 12345,
        "TransactionAmt": 299.99,
        "ProductCD": "W",
        "card1": 1234,
        "card4": "visa",
        "card6": "credit",
        "addr1": 315,
        "addr2": 87,
        "P_emaildomain": "gmail.com",
        "DeviceType": "mobile",
        "DeviceInfo": "iOS 15.0 / iPhone",
        "card1_txn_count_24hr": 8,
    }

    sample_decision = {
        "fraud_probability": 0.35,
        "action": "step_up",
        "reason": "Step-Up chosen: E[loss|step_up]=$3.25 vs E[loss|approve]=$25.00",
    }

    response = generate_dispute_response(sample_txn, sample_decision)
    print(response)
    print("\nDay 11 complete.")

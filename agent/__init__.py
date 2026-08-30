"""
Fraud Risk Agent.

A cost-sensitive fraud detection and decisioning system.
"""

__version__ = "1.0.0"

from .expected_loss import expected_loss, argmin_action
from .decision_agent import make_decision, run_agent_on_dataset

__all__ = ["expected_loss", "argmin_action", "make_decision", "run_agent_on_dataset", "__version__"]

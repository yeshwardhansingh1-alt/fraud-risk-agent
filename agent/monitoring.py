"""
Monitoring and Alerting Hooks (Day 26).

Calculates Population Stability Index (PSI) to monitor feature drift 
and prediction drift between the reference (training) distribution 
and a production (inference) window.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Calculate the Population Stability Index (PSI) for a continuous variable.
    """
    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = np.percentile(expected, breakpoints)
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Replace 0 to avoid division by zero and log(0)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return float(psi_value)


def check_prediction_drift(reference_probs: pd.Series, production_probs: pd.Series, threshold: float = 0.2):
    """
    Check if the distribution of predicted fraud probabilities has shifted significantly.
    """
    psi = calculate_psi(reference_probs.values, production_probs.values)
    
    if psi > threshold:
        logger.error(f"ALERT: Severe Prediction Drift Detected! PSI = {psi:.4f}")
        return True
    elif psi > threshold / 2:
        logger.warning(f"WARNING: Moderate Prediction Drift Detected. PSI = {psi:.4f}")
        return False
    
    logger.info(f"Predictions are stable. PSI = {psi:.4f}")
    return False


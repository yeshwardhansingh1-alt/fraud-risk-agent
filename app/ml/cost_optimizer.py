import numpy as np

class NetFinancialImpactOptimizer:
    def __init__(self, c_friction_pct=0.02, c_chargeback_fixed=15.0):
        """
        c_friction_pct: Lost lifetime value / merchant churn fee (2% of transaction).
        c_chargeback_fixed: Network dispute fee ($15 / ₹1200 per dispute).
        """
        self.c_friction_pct = c_friction_pct
        self.c_chargeback_fixed = c_chargeback_fixed

    def calculate_nfi(self, y_true: np.ndarray, y_probs: np.ndarray, amounts: np.ndarray, threshold: float) -> float:
        y_pred = (y_probs >= threshold).astype(int)
        
        # True Positives: Fraud blocked (Saved Amount - Chargeback Fee avoided)
        tp_mask = (y_true == 1) & (y_pred == 1)
        tp_savings = np.sum(amounts[tp_mask])
        
        # False Positives: Good transaction blocked (Lost Gateway Margin / Merchant Friction)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fp_cost = np.sum(amounts[fp_mask] * self.c_friction_pct)
        
        # False Negatives: Fraud missed (Gross Amount Lost + Fixed Chargeback Penalty)
        fn_mask = (y_true == 1) & (y_pred == 0)
        fn_loss = np.sum(amounts[fn_mask] + self.c_chargeback_fixed)
        
        return float(tp_savings - fp_cost - fn_loss)

    def find_optimal_threshold(self, y_true: np.ndarray, y_probs: np.ndarray, amounts: np.ndarray):
        thresholds = np.linspace(0.01, 0.99, 99)
        nfi_scores = [self.calculate_nfi(y_true, y_probs, amounts, t) for t in thresholds]
        best_idx = np.argmax(nfi_scores)
        return thresholds[best_idx], nfi_scores[best_idx]

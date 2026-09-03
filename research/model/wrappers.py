import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

class LGBMWrapper:
    """
    Wrap a LightGBM Booster to mimic the scikit-learn estimator interface.
    
    This wrapper ensures that we can call `.predict_proba(X)` on a raw 
    LightGBM booster, which is required by calibration routines like 
    CustomCalibratedClassifier.
    """

    def __init__(self, booster, feature_cols):
        self.booster = booster
        self.feature_cols = feature_cols
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols].values
        pos_prob = self.booster.predict(X)
        return np.column_stack([1 - pos_prob, pos_prob])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def fit(self, X, y):
        return self

    def get_params(self, deep=True):
        return {"booster": self.booster, "feature_cols": self.feature_cols}

    def set_params(self, **params):
        return self


class CustomCalibratedClassifier:
    """
    A custom wrapper for applying Isotonic Regression calibration 
    to any model wrapper that outputs uncalibrated probabilities.
    
    This is necessary because sklearn's CalibratedClassifierCV requires 
    models to implement the full sklearn estimator API, which raw LightGBM 
    Boosters do not perfectly support.
    """
    def __init__(self, model_wrapper):
        """
        Initialize the calibrator.
        
        Args:
            model_wrapper: Any object implementing `predict_proba(X)`.
        """
        self.model_wrapper = model_wrapper
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        
    def fit(self, X_val, y_val):
        y_pred_uncal = self.model_wrapper.predict_proba(X_val)[:, 1]
        self.calibrator.fit(y_pred_uncal, y_val)
        return self
        
    def predict_proba(self, X):
        y_pred_uncal = self.model_wrapper.predict_proba(X)[:, 1]
        y_pred_cal = self.calibrator.predict(y_pred_uncal)
        return np.column_stack([1 - y_pred_cal, y_pred_cal])
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

import pytest
import os
import joblib

def test_model_artifacts_exist():
    # Verify that the essential model artifacts have been produced
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    assert os.path.exists(os.path.join(model_dir, "lgbm_model.txt"))
    assert os.path.exists(os.path.join(model_dir, "calibrated_model.pkl"))
    assert os.path.exists(os.path.join(model_dir, "feature_cols.json"))
    assert os.path.exists(os.path.join(model_dir, "cost_config.json"))

def test_calibrated_model_loads():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    model = joblib.load(os.path.join(model_dir, "calibrated_model.pkl"))
    assert model is not None
    assert hasattr(model, "predict_proba")

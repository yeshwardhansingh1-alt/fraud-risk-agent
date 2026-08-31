import shap
import lightgbm as lgb
import pandas as pd
import numpy as np

class RealTimeScorer:
    def __init__(self, model_path: str):
        self.model = lgb.Booster(model_file=model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def predict_and_explain(self, feature_df: pd.DataFrame):
        # 1. Fast Probability Scoring
        prob = self.model.predict(feature_df)[0]
        
        # 2. Extract SHAP Local Vector
        shap_values = self.explainer.shap_values(feature_df)
        
        if isinstance(shap_values, list):
            # For multi-class/binary list outputs
            shap_values = shap_values[1]
            
        feature_names = feature_df.columns
        top_indices = np.argsort(np.abs(shap_values[0]))[::-1][:3]
        
        reasons = [
            {
                "feature": feature_names[i],
                "shap_value": float(shap_values[0][i]),
                "feature_value": float(feature_df.iloc[0, i])
            }
            for i in top_indices
        ]
        
        return prob, reasons

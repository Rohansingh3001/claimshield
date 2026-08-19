import shap
import pandas as pd
import numpy as np

class ModelExplainer:
    def __init__(self, model, X_train):
        """Initializes the SHAP explainer."""
        self.model = model
        # Using a sample of background data to keep it fast
        background = shap.sample(X_train, 100) if len(X_train) > 100 else X_train
        try:
            self.explainer = shap.Explainer(self.model, background)
        except Exception:
            # Fallback for models that might not support the general Explainer well
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            
    def get_explanation(self, X_instance):
        """Returns the SHAP values for a single instance."""
        if len(X_instance) == 0:
            return None
            
        shap_values = self.explainer(X_instance)
        
        # If it's a classifier that outputs probabilities, get the values for the positive class
        if len(shap_values.values.shape) == 3:
            values = shap_values.values[0, :, 1]
            base_value = shap_values.base_values[0, 1]
        else:
            values = shap_values.values[0]
            base_value = shap_values.base_values[0]
            
        feature_names = X_instance.columns.tolist()
        
        contributions = []
        for i in range(len(feature_names)):
            contributions.append({
                'Feature': feature_names[i],
                'Value': X_instance.iloc[0, i],
                'Contribution': values[i]
            })
            
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x['Contribution']), reverse=True)
        return contributions, base_value

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score, precision_score, confusion_matrix, roc_curve, precision_recall_curve

class ModelEvaluator:
    def __init__(self, models_dict):
        self.models = models_dict
        
    def evaluate(self, X_test, y_test):
        """Evaluates all models and returns a DataFrame of metrics."""
        results = []
        for name, model in self.models.items():
            # Get probability of positive class (Fraud)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'ROC-AUC': roc_auc,
                'PR-AUC': pr_auc,
                'Recall': recall,
                'Precision': precision,
                'F1 Score': f1
            })
        return pd.DataFrame(results)
        
    def get_curves(self, name, X_test, y_test):
        """Gets data for ROC and PR curves for a specific model."""
        if name not in self.models:
            raise ValueError(f"Model {name} not found.")
            
        model = self.models[name]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        prec, rec, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        return {
            'roc': {'fpr': fpr, 'tpr': tpr},
            'pr': {'precision': prec, 'recall': rec}
        }

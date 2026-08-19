import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json

class ModelTrainer:
    def __init__(self):
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        self.calibrated_models = {}
        
    def train_models(self, X_train, y_train):
        """Trains models with basic parameters and applies probability calibration."""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Calibrate probabilities
            print(f"Calibrating {name}...")
            calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            calibrated.fit(X_train, y_train)
            self.calibrated_models[name] = calibrated
            
    def save_models(self, models_dir):
        """Saves the trained models and their metadata."""
        import os
        os.makedirs(models_dir, exist_ok=True)
        for name, model in self.calibrated_models.items():
            joblib.dump(model, os.path.join(models_dir, f"{name.lower()}.pkl"))
            
        metadata = {
            "selected_model": "xgboost",
            "models_available": list(self.models.keys())
        }
        with open(os.path.join(models_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

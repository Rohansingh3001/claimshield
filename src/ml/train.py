import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
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
            
            # Hyperparameter tuning for XGBoost
            if name == 'XGBoost':
                print("Performing hyperparameter tuning for XGBoost...")
                param_grid = {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 4, 5, 6, 7],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 1, 5],            # Penalize complex trees
                    'reg_alpha': [0, 0.1, 1, 10],       # L1 regularization
                    'reg_lambda': [0, 1, 10, 50],       # L2 regularization
                    'scale_pos_weight': [1, 2, 5, 10]   # Boost recall for positive class
                }
                search = RandomizedSearchCV(
                    model, 
                    param_distributions=param_grid, 
                    n_iter=20, 
                    scoring='recall', 
                    cv=5, 
                    random_state=42, 
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                print(f"Best parameters found: {search.best_params_}")
            elif name == 'RandomForest':
                print("Performing hyperparameter tuning for RandomForest...")
                rf_param_grid = {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }
                rf_search = RandomizedSearchCV(
                    model, 
                    param_distributions=rf_param_grid, 
                    n_iter=20, 
                    scoring='recall', 
                    cv=5, 
                    random_state=42, 
                    n_jobs=-1
                )
                rf_search.fit(X_train, y_train)
                model = rf_search.best_estimator_
                print(f"Best parameters found: {rf_search.best_params_}")
            else:
                model.fit(X_train, y_train)
            
            # Calibrate probabilities
            print(f"Calibrating {name}...")
            calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
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

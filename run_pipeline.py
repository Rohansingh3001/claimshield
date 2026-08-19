import os
import sys
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.features.engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.ml.train import ModelTrainer
from src.ml.evaluate import ModelEvaluator
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    print("Starting ML Pipeline...")
    
    # 1. Load Data
    data_path = "data/sample/ClaimShieldAI-Dataset.csv"
    print(f"Loading data from {data_path}...")
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    # 2. Validation
    print("Validating data...")
    validator = DataValidator(df)
    report = validator.validate()
    print("Validation Report:", {k: v for k, v in report.items() if k != 'missing_values'})
    leakage_cols = report['potential_leakage']
    print(f"Identified potential leakage columns: {leakage_cols}")
    
    # 3. Feature Engineering
    print("Engineering features...")
    engineer = FeatureEngineer(df)
    df_engineered = engineer.engineer_features()
    
    # 4. Preprocessing & Split
    print("Preprocessing data...")
    preprocessor = DataPreprocessor(target_col='Fraud_Flag')
    X, y = preprocessor.prepare_data(df_engineered, leakage_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    os.makedirs('models', exist_ok=True)
    preprocessor.save('models/preprocessor.pkl')
    
    # 5. Training
    print("Training models...")
    trainer = ModelTrainer()
    trainer.train_models(X_train_processed, y_train)
    trainer.save_models('models')
    
    # 6. Evaluation
    print("Evaluating models...")
    # Evaluate calibrated models
    evaluator = ModelEvaluator(trainer.calibrated_models)
    results = evaluator.evaluate(X_test_processed, y_test)
    print("\nEvaluation Results:")
    print(results)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()

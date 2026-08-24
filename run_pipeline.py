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
import json
from imblearn.over_sampling import SMOTE

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
    # Split: 60% Train, 20% Validation, 20% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Data split - Train: {len(X_train)} (60%), Validation: {len(X_val)} (20%), Test: {len(X_test)} (20%)")
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    os.makedirs('models', exist_ok=True)
    preprocessor.save('models/preprocessor.pkl')
    
    # 5. Training
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    print("Training models...")
    trainer = ModelTrainer()
    trainer.train_models(X_train_resampled, y_train_resampled)
    trainer.save_models('models')
    
    # 6. Evaluation
    print("Evaluating models...")
    # Evaluate calibrated models
    evaluator = ModelEvaluator(trainer.calibrated_models)
    
    print("\nValidation Set Results (20%):")
    val_results = evaluator.evaluate(X_val_processed, y_val)
    print(val_results)
    
    print("\nTest Set Results (20%):")
    test_results = evaluator.evaluate(X_test_processed, y_test)
    print(test_results)
    
    # Save the full model comparison table
    test_results.to_csv('models/model_comparison.csv', index=False)
    
    # Save metrics to JSON for the dashboard
    xgboost_metrics = test_results[test_results['Model'] == 'XGBoost'].to_dict('records')[0]
    metrics_out = {
        'Accuracy': float(xgboost_metrics.get('Accuracy', 0)),
        'ROC-AUC': float(xgboost_metrics.get('ROC-AUC', 0)),
        'PR-AUC': float(xgboost_metrics.get('PR-AUC', 0)),
        'Recall': float(xgboost_metrics.get('Recall', 0)),
        'Precision': float(xgboost_metrics.get('Precision', 0)),
        'F1 Score': float(xgboost_metrics.get('F1 Score', 0)),
        'TN': int(xgboost_metrics.get('TN', 0)),
        'FP': int(xgboost_metrics.get('FP', 0)),
        'FN': int(xgboost_metrics.get('FN', 0)),
        'TP': int(xgboost_metrics.get('TP', 0))
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=4)
        
    # Generate and save curve data for XGBoost
    curves = evaluator.get_curves('XGBoost', X_test_processed, y_test)
    curves_out = {
        'roc': {
            'fpr': curves['roc']['fpr'].tolist(),
            'tpr': curves['roc']['tpr'].tolist()
        },
        'pr': {
            'precision': curves['pr']['precision'].tolist(),
            'recall': curves['pr']['recall'].tolist()
        }
    }
    with open('models/curves.json', 'w') as f:
        json.dump(curves_out, f)
        
    # Generate Test Validation Results (Ground Truth vs Prediction)
    print("Generating Prediction vs Ground Truth validation table...")
    final_model = trainer.calibrated_models['XGBoost']
    test_probs = final_model.predict_proba(X_test_processed)[:, 1]
    
    # In evaluate.py, we use a custom threshold of 0.35
    test_preds = (test_probs >= 0.35).astype(int)
    
    # Get original Claim_ID for test set using index
    test_validation_df = pd.DataFrame({
        'Claim_ID': df.loc[X_test.index, 'Claim_ID'] if 'Claim_ID' in df.columns else X_test.index,
        'Actual Label': df.loc[X_test.index, 'Fraud_Flag'],
        'Fraud Probability': (test_probs * 100).round(2),
        'Predicted Label': test_preds
    })
    
    # Map predictions to string labels
    test_validation_df['Predicted Label'] = test_validation_df['Predicted Label'].apply(lambda x: 'Fraud' if x == 1 else 'Normal')
    
    # Compare
    def check_correct(row):
        actual_is_fraud = str(row['Actual Label']).lower() in ['yes', '1', 'true', 'fraud']
        predicted_is_fraud = row['Predicted Label'] == 'Fraud'
        return 'Correct' if actual_is_fraud == predicted_is_fraud else 'Incorrect'
        
    test_validation_df['Result'] = test_validation_df.apply(check_correct, axis=1)
    
    # Save validation table
    test_validation_df.to_csv('data/sample/Test-Validation-Results.csv', index=False)
        
    # Batch Inference on the entire dataset for the dashboard
    print("Running batch inference on entire dataset to generate Scored-Dataset.csv...")
    X_all_processed = preprocessor.transform(X)
    final_model = trainer.calibrated_models['XGBoost']
    fraud_probs = final_model.predict_proba(X_all_processed)[:, 1]
    
    df_scored = df.copy()
    df_scored['Fraud_Risk_Score'] = (fraud_probs * 100).round(2)
    df_scored.to_csv('data/sample/Scored-Dataset.csv', index=False)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()

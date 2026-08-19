import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

class DataPreprocessor:
    def __init__(self, target_col='Fraud_Flag'):
        self.target_col = target_col
        self.preprocessor = None
        self.feature_names_out_ = None
        
    def prepare_data(self, df, leakage_cols):
        """Drops leakage columns and separates X and y."""
        drop_cols = leakage_cols + [self.target_col]
        # Drop identifier columns too
        id_cols = [c for c in df.columns if 'id' in c.lower() or 'number' in c.lower()]
        drop_cols.extend(id_cols)
        
        drop_cols = list(set([col for col in drop_cols if col in df.columns]))
        
        X = df.drop(columns=drop_cols)
        # Assuming target is categorical Yes/No or 1/0
        y = df[self.target_col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true', 'fraud'] else 0)
        
        return X, y
        
    def build_pipeline(self, X):
        """Builds the preprocessing pipeline based on column types."""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        return self.preprocessor

    def fit_transform(self, X):
        """Fits and transforms the data, and saves the pipeline."""
        if self.preprocessor is None:
            self.build_pipeline(X)
            
        X_processed = self.preprocessor.fit_transform(X)
        self.feature_names_out_ = self.preprocessor.get_feature_names_out()
        return pd.DataFrame(X_processed, columns=self.feature_names_out_, index=X.index)

    def transform(self, X):
        """Transforms data using the fitted pipeline."""
        X_processed = self.preprocessor.transform(X)
        return pd.DataFrame(X_processed, columns=self.feature_names_out_, index=X.index)
        
    def save(self, filepath):
        """Saves the fitted preprocessor."""
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath):
        """Loads a saved preprocessor."""
        return joblib.load(filepath)

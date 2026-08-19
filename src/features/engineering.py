import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_ratio_features(self):
        """Creates financial ratio features if underlying columns exist."""
        # Claim to Coverage Ratio
        if 'Claim_Amount' in self.df.columns and 'Coverage_Amount' in self.df.columns:
            self.df['claim_to_coverage_ratio'] = self.df['Claim_Amount'] / (self.df['Coverage_Amount'] + 1e-5)
            
        # Premium to Coverage Ratio
        if 'Premium_Amount' in self.df.columns and 'Coverage_Amount' in self.df.columns:
            self.df['premium_to_coverage_ratio'] = self.df['Premium_Amount'] / (self.df['Coverage_Amount'] + 1e-5)
            
        return self.df

    def process_dates(self):
        """Extracts date features if any date columns exist."""
        date_cols = self.df.select_dtypes(include=['datetime64', 'object']).columns
        # For simplicity in MVP, we just look for common date column names or assume they are parsed elsewhere.
        # This dataset might not have dates, so we skip if none are obvious datetime objects.
        return self.df

    def engineer_features(self):
        """Runs the full feature engineering pipeline."""
        self.df = self.create_ratio_features()
        self.df = self.process_dates()
        return self.df

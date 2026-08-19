import pandas as pd
import numpy as np

class DataValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Potential leakage columns based on typical insurance dataset post-claim events
        self.leakage_keywords = ['status', 'settlement', 'outcome', 'decision', 'post', 'risk_score', 'fraud_score']

    def check_missing_values(self):
        """Returns a series of missing value percentages per column."""
        return (self.df.isnull().sum() / len(self.df)) * 100

    def check_duplicates(self):
        """Returns the number of duplicate rows."""
        return self.df.duplicated().sum()

    def identify_leakage_columns(self, target_col='Fraud_Flag'):
        """Identifies potential data leakage columns."""
        leakage_cols = []
        for col in self.df.columns:
            if col == target_col:
                continue
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in self.leakage_keywords):
                leakage_cols.append(col)
        return leakage_cols

    def validate(self):
        """Runs basic validation and returns a summary dictionary."""
        validation_report = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.check_missing_values().to_dict(),
            'duplicates': self.check_duplicates(),
            'potential_leakage': self.identify_leakage_columns()
        }
        return validation_report

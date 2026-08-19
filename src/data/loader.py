import pandas as pd
import os
import json

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
    
    def load_data(self):
        """Loads data from CSV or Excel depending on the extension."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset not found at {self.filepath}")
            
        ext = os.path.splitext(self.filepath)[-1].lower()
        if ext == '.csv':
            return pd.read_csv(self.filepath)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(self.filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

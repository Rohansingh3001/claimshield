import pandas as pd

class RiskScorer:
    def __init__(self):
        pass
        
    def calculate_risk_score(self, fraud_probability):
        """Converts fraud probability to a 0-100 scale."""
        return int(round(fraud_probability * 100))

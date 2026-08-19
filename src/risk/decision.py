class DecisionEngine:
    def __init__(self, low_threshold=30, high_threshold=70):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
    def get_recommendation(self, risk_score):
        """Returns Approve, Review, or Investigate based on risk score."""
        if risk_score <= self.low_threshold:
            return "APPROVE"
        elif risk_score <= self.high_threshold:
            return "REVIEW"
        else:
            return "INVESTIGATE"
            
    def calculate_exposure(self, claim_amount, fraud_probability):
        """Estimates potential financial exposure."""
        if claim_amount is None or pd.isna(claim_amount):
            return 0.0
        return float(claim_amount) * float(fraud_probability)

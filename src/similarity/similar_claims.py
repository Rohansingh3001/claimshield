import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np

class SimilarityEngine:
    def __init__(self, historical_df, features_to_use=None):
        self.historical_df = historical_df.copy()
        
        # Use numerical features for simplicity if not specified
        if features_to_use is None:
            self.features = self.historical_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            # Remove any leakage columns or target from similarity if present
            self.features = [f for f in self.features if f not in ['Fraud_Risk_Score', 'Settlement_Amount']]
        else:
            self.features = features_to_use
            
        self.scaler = StandardScaler()
        self.knn = NearestNeighbors(n_neighbors=6, metric='cosine') # 6 because 1 is the claim itself if it's in the dataset
        
    def fit(self):
        """Fits the similarity engine on historical data."""
        X = self.historical_df[self.features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.knn.fit(X_scaled)
        
    def find_similar(self, query_instance, top_n=5):
        """Finds top N similar claims."""
        # Ensure query instance has the same features
        X_query = query_instance[self.features].fillna(0)
        X_query_scaled = self.scaler.transform(X_query)
        
        distances, indices = self.knn.kneighbors(X_query_scaled, n_neighbors=top_n + 1)
        
        # Exclude the first one if distance is 0 (it's the exact same claim)
        # For a new claim not in the set, the first one is just the closest.
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            if dist < 1e-5 and i == 0:
                continue # Skip exact match
                
            similarity_score = int(round((1 - dist) * 100)) # Cosine distance to similarity
            similar_claim = self.historical_df.iloc[idx]
            
            outcome = similar_claim.get('Fraud_Flag', 'Unknown')
            claim_id = similar_claim.get('Claim_ID', f"Index_{idx}")
            
            results.append({
                'Claim_ID': claim_id,
                'Similarity': f"{similarity_score}%",
                'Historical Outcome': outcome
            })
            
            if len(results) == top_n:
                break
                
        return pd.DataFrame(results)

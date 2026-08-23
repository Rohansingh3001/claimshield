import pandas as pd
import numpy as np

def inject_probabilistic_signal(csv_path):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Calculate a continuous risk probability for each row
    np.random.seed(42)
    
    # Base probability
    prob = np.full(len(df), 0.02)
    
    # 1. Claim to Coverage Ratio
    claim_ratio = (df['Claim_Amount'] / (df['Coverage_Amount'] + 1)).clip(0, 1)
    prob += 0.35 * claim_ratio
    
    # 2. Credit Score (Lower is riskier)
    # Credit ranges 300-850. 
    credit_factor = (850 - df['Credit_Score'].clip(300, 850)) / 550
    prob += 0.25 * credit_factor
    
    # 3. Age (Younger is slightly riskier for some claims)
    age_factor = ((40 - df['Age']).clip(0, 40)) / 22  # e.g., age 18 -> 1, age 40 -> 0
    prob += 0.08 * age_factor
    
    # 4. Claim Type
    prob += np.where(df['Claim_Type'].isin(['Theft', 'Fire']), 0.15, 0.0)
    
    # 5. Mismatches (Highly suspicious)
    mismatch1 = (df['Policy_Type'] == 'Home') & (df['Claim_Type'] == 'Medical')
    prob += np.where(mismatch1, 0.4, 0.0)
    
    mismatch2 = (df['Policy_Type'] == 'Health') & (df['Claim_Type'].isin(['Collision', 'Property Damage', 'Fire', 'Theft']))
    prob += np.where(mismatch2, 0.4, 0.0)
    
    # Clip probabilities to keep them realistic [0.01, 0.98]
    prob = np.clip(prob, 0.01, 0.98)
    
    # Sample Fraud_Flag using binomial distribution
    fraud_binary = np.random.binomial(1, prob)
    df['Fraud_Flag'] = np.where(fraud_binary == 1, 'Yes', 'No')
    
    fraud_count = len(df[df['Fraud_Flag'] == 'Yes'])
    print(f"Total Fraud Cases Injected: {fraud_count} ({(fraud_count/len(df))*100:.2f}%)")
    
    df.to_csv(csv_path, index=False)
    print(f"Saved probabilistically augmented dataset back to {csv_path}")

if __name__ == '__main__':
    inject_probabilistic_signal('data/sample/ClaimShieldAI-Dataset.csv')

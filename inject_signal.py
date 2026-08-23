import pandas as pd
import numpy as np

def inject_fraud_signal(csv_path):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Initialize with No
    df['Fraud_Flag'] = 'No'
    
    # Rule 1: High claim relative to coverage AND low credit score
    rule1 = (df['Claim_Amount'] > 0.8 * df['Coverage_Amount']) & (df['Credit_Score'] < 550)
    
    # Rule 2: Theft/Fire claims for young people with decently high amounts
    rule2 = (df['Claim_Type'].isin(['Theft', 'Fire'])) & (df['Age'] < 30) & (df['Claim_Amount'] > 0.4 * df['Coverage_Amount'])
    
    # Rule 3: Mismatches
    rule3 = (df['Policy_Type'] == 'Home') & (df['Claim_Type'] == 'Medical')
    rule4 = (df['Policy_Type'] == 'Health') & (df['Claim_Type'].isin(['Collision', 'Property Damage', 'Fire', 'Theft']))
    
    # Apply rules
    is_fraud = rule1 | rule2 | rule3 | rule4
    df.loc[is_fraud, 'Fraud_Flag'] = 'Yes'
    
    # Add 3% random noise
    np.random.seed(42)
    flip_mask = np.random.rand(len(df)) < 0.03
    
    def flip_flag(flag):
        return 'No' if flag == 'Yes' else 'Yes'
        
    df.loc[flip_mask, 'Fraud_Flag'] = df.loc[flip_mask, 'Fraud_Flag'].apply(flip_flag)
    
    fraud_count = len(df[df['Fraud_Flag'] == 'Yes'])
    print(f"Total Fraud Cases Injected: {fraud_count} ({(fraud_count/len(df))*100:.2f}%)")
    
    df.to_csv(csv_path, index=False)
    print(f"Saved augmented dataset back to {csv_path}")

if __name__ == '__main__':
    inject_fraud_signal('data/sample/ClaimShieldAI-Dataset.csv')

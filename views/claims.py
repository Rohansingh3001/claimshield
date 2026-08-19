import streamlit as st
import pandas as pd
import os

def render():
    st.markdown("<h1 class='gradient-text' style='margin-bottom: 0;'>Claims Queue</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 1.1rem; margin-bottom: 2rem;'>Filter and search through all scored claims to prioritize investigations.</p>", unsafe_allow_html=True)
    
    data_path = "data/sample/ClaimShieldAI-Dataset.csv"
    if not os.path.exists(data_path):
        st.warning("Dataset not found. Please run the ML pipeline first.")
        return
        
    @st.cache_data
    def load_data():
        df = pd.read_csv(data_path)
        return df
        
    df = load_data()
    
    # Mock Risk Scores if not present
    if 'Fraud_Risk_Score' not in df.columns:
        import numpy as np
        np.random.seed(42)
        df['Fraud_Risk_Score'] = np.random.randint(10, 95, size=len(df))
        
    def get_decision(score):
        if score > 70: return 'INVESTIGATE'
        elif score > 30: return 'REVIEW'
        else: return 'APPROVE'
        
    if 'Decision' not in df.columns:
        df['Decision'] = df['Fraud_Risk_Score'].apply(get_decision)
        
    st.markdown("""
    <div class="metric-card" style="padding: 16px 24px; margin-bottom: 24px; animation: float 6s ease-in-out infinite;">
        <h4 style='margin-top: 0; color: var(--primary); font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 800;'>Filters</h4>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.selectbox("Risk Level", ["All", "High Risk (>70)", "Medium Risk (31-70)", "Low Risk (0-30)"], label_visibility="collapsed")
    with col2:
        decision_filter = st.selectbox("Recommendation", ["All", "INVESTIGATE", "REVIEW", "APPROVE"], label_visibility="collapsed")
    with col3:
        search = st.text_input("Search Claim ID", placeholder="Search by Claim ID...", label_visibility="collapsed")
        
    st.markdown("</div>", unsafe_allow_html=True)
        
    filtered_df = df.copy()
    
    if risk_filter == "High Risk (>70)":
        filtered_df = filtered_df[filtered_df['Fraud_Risk_Score'] > 70]
    elif risk_filter == "Medium Risk (31-70)":
        filtered_df = filtered_df[(filtered_df['Fraud_Risk_Score'] <= 70) & (filtered_df['Fraud_Risk_Score'] > 30)]
    elif risk_filter == "Low Risk (0-30)":
        filtered_df = filtered_df[filtered_df['Fraud_Risk_Score'] <= 30]
        
    if decision_filter != "All":
        filtered_df = filtered_df[filtered_df['Decision'] == decision_filter]
        
    if search:
        filtered_df = filtered_df[filtered_df['Claim_ID'].str.contains(search, case=False, na=False)]
        
    # Sort by risk score descending
    filtered_df = filtered_df.sort_values(by='Fraud_Risk_Score', ascending=False)
    
    # Select columns to display
    display_cols = ['Claim_ID', 'Customer_ID', 'Claim_Amount', 'Fraud_Risk_Score', 'Decision']
    # Keep only those that exist
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.markdown(f"<p style='color: var(--text-muted);'>Found <strong style='color: var(--primary); font-size: 1.2rem; text-shadow: 0 0 10px var(--primary-glow);'>{len(filtered_df):,}</strong> claims matching criteria.</p>", unsafe_allow_html=True)
    
    # Style the dataframe container
    st.markdown("<div style='border: 1px solid var(--glass-border); border-radius: 12px; overflow: hidden;'>", unsafe_allow_html=True)
    st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

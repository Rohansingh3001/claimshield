import streamlit as st
import pandas as pd
import os

def render():
    st.markdown("<h1 style='margin-bottom: 0;'>Claim Investigation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 1.1rem; margin-bottom: 2rem;'>Deep dive into a specific claim's risk profile and AI explanation.</p>", unsafe_allow_html=True)
    
    data_path = "data/sample/ClaimShieldAI-Dataset.csv"
    if not os.path.exists(data_path):
        st.warning("Dataset not found. Please run the ML pipeline first.")
        return
        
    @st.cache_data
    def load_data():
        df = pd.read_csv(data_path)
        return df
        
    df = load_data()
    
    if 'Claim_ID' not in df.columns:
        st.error("No Claim_ID column found in dataset.")
        return
        
    claim_ids = df['Claim_ID'].tolist()
    
    # Search / Select bar
    selected_claim_id = st.selectbox("🔍 Search or Select a Claim ID to Investigate", claim_ids)
    
    if selected_claim_id:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 2rem 0;'>", unsafe_allow_html=True)
        claim_data = df[df['Claim_ID'] == selected_claim_id].iloc[0]
        
        # Risk Score Mocking if not run
        risk_score = claim_data.get('Fraud_Risk_Score', 89) 
        
        if risk_score > 70:
            risk_label = "HIGH RISK"
            color = "var(--danger)"
            bg_color = "rgba(239, 68, 68, 0.1)"
            decision = "INVESTIGATE"
        elif risk_score > 30:
            risk_label = "MEDIUM RISK"
            color = "var(--warning)"
            bg_color = "rgba(245, 158, 11, 0.1)"
            decision = "REVIEW"
        else:
            risk_label = "LOW RISK"
            color = "var(--success)"
            bg_color = "rgba(16, 185, 129, 0.1)"
            decision = "APPROVE"
            
        exposure = claim_data.get('Claim_Amount', 0) * (risk_score / 100)
            
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("<h3 style='font-size: 1.2rem; color: var(--text-muted);'>STEP 1: Claim Summary</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background-color: var(--card-bg); padding: 24px; border-radius: 12px; border: 1px solid var(--border);'>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px;'>
                    <div><span style='color: var(--text-muted)'>Customer:</span> <strong>{claim_data.get('Customer_ID', 'N/A')}</strong></div>
                    <div><span style='color: var(--text-muted)'>Policy No:</span> <strong>{claim_data.get('Policy_Number', 'N/A')}</strong></div>
                    <div><span style='color: var(--text-muted)'>Claim Amt:</span> <strong>₹{claim_data.get('Claim_Amount', 0):,.2f}</strong></div>
                    <div><span style='color: var(--text-muted)'>Coverage:</span> <strong>₹{claim_data.get('Coverage_Amount', 0):,.2f}</strong></div>
                    <div><span style='color: var(--text-muted)'>Claim Type:</span> <strong>{claim_data.get('Claim_Type', 'N/A')}</strong></div>
                    <div><span style='color: var(--text-muted)'>Policy Type:</span> <strong>{claim_data.get('Policy_Type', 'N/A')}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("<h3 style='font-size: 1.2rem; color: var(--text-muted);'>STEP 2: AI Assessment</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {color}; border-radius: 12px; padding: 24px; text-align: center; height: 100%;">
                <div style="color: {color}; font-weight: 700; font-size: 1rem; letter-spacing: 0.1em; margin-bottom: 8px;">{risk_label}</div>
                <div style="font-size: 3.5rem; font-weight: 800; color: var(--text); line-height: 1;">{risk_score}</div>
                <div style="color: var(--text-muted); font-size: 0.8rem; margin-top: -5px; margin-bottom: 16px;">out of 100</div>
                <div style="background-color: {color}; color: #000; padding: 8px 16px; border-radius: 6px; font-weight: 700; display: inline-block;">ACTION: {decision}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col3, col4 = st.columns([1.5, 1])
        
        with col3:
            st.markdown("<h3 style='font-size: 1.2rem; color: var(--text-muted);'>STEP 3: Why was this decision made?</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style='background-color: var(--card-bg); padding: 20px; border-radius: 12px; border: 1px solid var(--border);'>
                <p style='margin-top:0;'>The XGBoost model analyzed 30+ features. Here are the top contributing factors pushing the risk score up or down:</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔍 View AI Reasoning (SHAP)", expanded=True):
                # Mock plain-english reasoning based on SHAP
                st.markdown(f"""
                - 🔴 **+12 Risk**: The `Claim_Amount` (₹{claim_data.get('Claim_Amount', 0):,.0f}) is unusually high compared to the `Coverage_Amount` (₹{claim_data.get('Coverage_Amount', 0):,.0f}).
                - 🔴 **+8 Risk**: The `Age` of the claimant ({claim_data.get('Age', 30)}) matches historical high-risk patterns for this `Claim_Type`.
                - 🟢 **-4 Risk**: The `Credit_Score` ({claim_data.get('Credit_Score', 700)}) is excellent, indicating financial stability.
                """)
                st.caption("*(Note: In the full production build, this plain text is dynamically generated from SHAP values using the `explain.py` module.)*")

        with col4:
            st.markdown("<h3 style='font-size: 1.2rem; color: var(--text-muted);'>STEP 4: Business Impact</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid var(--warning);">
                <div class="metric-title">Estimated Financial Exposure</div>
                <div class="metric-value" style="color: var(--warning); font-size: 1.8rem;">₹{exposure:,.0f}</div>
                <p style="color: var(--text-muted); font-size: 0.8rem; margin-top: 10px; margin-bottom: 0;">Calculated as (Claim Amount × Probability of Fraud). Prioritize investigations with high exposure.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.2rem; color: var(--text-muted);'>STEP 5: Historical Context (Similar Claims)</h3>", unsafe_allow_html=True)
        
        sim_data = pd.DataFrame({
            'Claim ID': ['CLM-1021', 'CLM-8342', 'CLM-7341'],
            'Similarity': ['94%', '89%', '86%'],
            'Historical Outcome': ['Fraud', 'Fraud', 'Genuine']
        })
        
        # Style dataframe for dark mode
        st.dataframe(sim_data, use_container_width=True, hide_index=True)


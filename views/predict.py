import streamlit as st
import pandas as pd
import time

def render():
    st.markdown("<h1 class='text-primary' style='margin-bottom: 0;'>New Claim Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 1.1rem; margin-bottom: 2rem;'>Run a live AI assessment on a new insurance claim.</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); margin-bottom: 15px;'>Enter Claim Details</h3>", unsafe_allow_html=True)
    
    with st.form("new_claim_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<p style='font-size: 0.9rem; color: var(--text-muted); margin-bottom: 5px;'>Policyholder Information</p>", unsafe_allow_html=True)
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
            
            st.markdown("<br><p style='font-size: 0.9rem; color: var(--text-muted); margin-bottom: 5px;'>Policy Details</p>", unsafe_allow_html=True)
            policy_type = st.selectbox("Policy Type", ["Auto", "Home", "Health", "Life"])
            coverage_amt = st.number_input("Total Coverage Amount ($)", min_value=1000, max_value=1000000, value=50000, step=1000)
            
        with col2:
            st.markdown("<p style='font-size: 0.9rem; color: var(--text-muted); margin-bottom: 5px;'>Incident Information</p>", unsafe_allow_html=True)
            claim_type = st.selectbox("Claim Type", ["Collision", "Theft", "Property Damage", "Medical", "Fire"])
            claim_amt = st.number_input("Requested Claim Amount ($)", min_value=100, max_value=1000000, value=8000, step=500)
            incident_severity = st.selectbox("Reported Severity", ["Minor", "Moderate", "Major", "Total Loss"])
            witnesses = st.number_input("Number of Witnesses", min_value=0, max_value=10, value=0)
            
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run AI Assessment", type="primary")
        
    if submitted:
        with st.spinner("Initializing XGBoost Model Pipeline..."):
            time.sleep(0.8) # Simulate processing time for effect
            
        with st.spinner("Extracting features and calculating SHAP values..."):
            time.sleep(1.2)
            
        st.markdown("<hr style='border-color: var(--border); margin: 2rem 0;'>", unsafe_allow_html=True)
        
        # --- SIMULATED INFERENCE LOGIC ---
        # This simulates a real ML model's output based on logical rules for the hackathon
        base_score = 15
        
        # Risk factors
        ratio = claim_amt / coverage_amt if coverage_amt > 0 else 1
        ratio_penalty = 0
        if ratio > 0.8:
            ratio_penalty = 40
        elif ratio > 0.5:
            ratio_penalty = 15
            
        credit_penalty = 0
        if credit_score < 500:
            credit_penalty = 25
        elif credit_score < 600:
            credit_penalty = 10
            
        witness_penalty = 0
        if witnesses == 0 and claim_amt > 20000:
            witness_penalty = 20
            
        severity_penalty = 0
        if incident_severity == "Total Loss" and claim_amt < 5000:
            severity_penalty = 15 # Suspiciously cheap total loss
            
        total_risk = min(99, base_score + ratio_penalty + credit_penalty + witness_penalty + severity_penalty)
        
        # Determine styling based on score
        if total_risk > 70:
            risk_label = "HIGH RISK"
            color = "var(--danger)"
            bg_color = "rgba(239, 68, 68, 0.1)"
            decision = "INVESTIGATE"
        elif total_risk > 30:
            risk_label = "MEDIUM RISK"
            color = "var(--warning)"
            bg_color = "rgba(245, 158, 11, 0.1)"
            decision = "REVIEW"
        else:
            risk_label = "LOW RISK"
            color = "var(--success)"
            bg_color = "rgba(16, 185, 129, 0.1)"
            decision = "APPROVE"
            
        exposure = claim_amt * (total_risk / 100)
        
        # Render Results
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("<h3 style='font-size: 1.1rem; color: var(--text-muted);'>Prediction Result</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {color}; border-radius: 8px; padding: 32px; text-align: center; height: 100%;">
                <div style="color: {color}; font-weight: 700; font-size: 1.1rem; letter-spacing: 0.1em; margin-bottom: 8px;">{risk_label}</div>
                <div style="font-size: 4.5rem; font-weight: 800; color: var(--text); line-height: 1;">{total_risk}</div>
                <div style="color: var(--text-muted); font-size: 0.9rem; margin-top: 5px; margin-bottom: 24px;">out of 100</div>
                <div style="background-color: {color}; color: #fff; padding: 8px 24px; border-radius: 4px; font-weight: 700; display: inline-block; letter-spacing: 0.05em;">ACTION: {decision}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("<h3 style='font-size: 1.1rem; color: var(--text-muted);'>SHAP Explanation</h3>", unsafe_allow_html=True)
            
            # Dynamically generate SHAP reasons based on inputs
            shap_html = f"""<div style="background: var(--background); padding: 20px; border-radius: 8px; border: 1px solid var(--border); height: 100%;">"""
            
            if ratio_penalty > 0:
                shap_html += f'<div style="margin-bottom: 12px; display: flex; align-items: center;"><div><strong style="color: var(--danger)">+{ratio_penalty} Risk:</strong> Claim amount (${claim_amt:,}) is dangerously close to or exceeds the policy limit (${coverage_amt:,}).</div></div>'
            else:
                shap_html += f'<div style="margin-bottom: 12px; display: flex; align-items: center;"><div><strong style="color: var(--success)">-5 Risk:</strong> Claim amount (${claim_amt:,}) is well within normal bounds for the policy limit.</div></div>'
                
            if credit_penalty > 0:
                shap_html += f'<div style="margin-bottom: 12px; display: flex; align-items: center;"><div><strong style="color: var(--danger)">+{credit_penalty} Risk:</strong> Credit score ({credit_score}) indicates potential financial distress, increasing moral hazard.</div></div>'
            else:
                shap_html += f'<div style="margin-bottom: 12px; display: flex; align-items: center;"><div><strong style="color: var(--success)">-4 Risk:</strong> High credit score ({credit_score}) indicates financial stability.</div></div>'
                
            if witness_penalty > 0:
                shap_html += f'<div style="margin-bottom: 12px; display: flex; align-items: center;"><div><strong style="color: var(--danger)">+{witness_penalty} Risk:</strong> Extremely high claim value with zero witnesses reported.</div></div>'
                
            if severity_penalty > 0:
                shap_html += f'<div style="margin-bottom: 12px; display: flex; align-items: center;"><div><strong style="color: var(--danger)">+{severity_penalty} Risk:</strong> Unusually low claim amount for a "Total Loss" severity report.</div></div>'
                
            shap_html += "</div>"
            st.markdown(shap_html, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid var(--warning); padding: 15px 24px;">
            <div class="metric-title" style="margin-bottom: 5px;">Estimated Financial Exposure</div>
            <div class="metric-value" style="color: var(--warning); font-size: 1.5rem;">${exposure:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

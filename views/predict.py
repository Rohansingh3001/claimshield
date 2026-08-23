import streamlit as st
import pandas as pd
import time
import joblib
import os
from src.features.engineering import FeatureEngineer
from src.ml.explain import ModelExplainer

def render():
    st.markdown("<h1 class='text-primary' style='margin-bottom: 0;'>New Claim Validator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 1.1rem; margin-bottom: 2rem;'>Run a live AI assessment on a new insurance claim using the trained XGBoost model.</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); margin-bottom: 15px;'>Enter Claim Details</h3>", unsafe_allow_html=True)
    
    with st.form("new_claim_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<p style='font-size: 0.9rem; color: var(--text-muted); margin-bottom: 5px;'>Policyholder Information</p>", unsafe_allow_html=True)
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            region = st.selectbox("Region", ["North", "South", "East", "West"])
            credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
            
            st.markdown("<br><p style='font-size: 0.9rem; color: var(--text-muted); margin-bottom: 5px;'>Policy Details</p>", unsafe_allow_html=True)
            policy_type = st.selectbox("Policy Type", ["Auto", "Home", "Health", "Life"])
            coverage_amt = st.number_input("Total Coverage Amount ($)", min_value=1000, max_value=1000000, value=50000, step=1000)
            premium_amt = st.number_input("Premium Amount ($)", min_value=100, max_value=50000, value=1500, step=100)
            
        with col2:
            st.markdown("<p style='font-size: 0.9rem; color: var(--text-muted); margin-bottom: 5px;'>Incident Information</p>", unsafe_allow_html=True)
            claim_type = st.selectbox("Claim Type", ["Collision", "Theft", "Property Damage", "Medical", "Fire", "Accident"])
            claim_amt = st.number_input("Requested Claim Amount ($)", min_value=100, max_value=1000000, value=8000, step=500)
            
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run AI Assessment", type="primary")
        
    if submitted:
        if not os.path.exists("models/preprocessor.pkl") or not os.path.exists("models/xgboost.pkl"):
            st.error("Model artifacts not found! Please run the ML pipeline first.")
            return

        with st.spinner("Loading XGBoost Model Pipeline..."):
            preprocessor = joblib.load("models/preprocessor.pkl")
            model = joblib.load("models/xgboost.pkl")
            
        with st.spinner("Extracting features and calculating SHAP values..."):
            # Prepare input data
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Region': [region],
                'Credit_Score': [credit_score],
                'Policy_Type': [policy_type],
                'Coverage_Amount': [coverage_amt],
                'Premium_Amount': [premium_amt],
                'Claim_Type': [claim_type],
                'Claim_Amount': [claim_amt]
            })
            
            # Feature Engineering
            engineer = FeatureEngineer(input_data)
            df_engineered = engineer.engineer_features()
            
            # Preprocessing
            X_processed = preprocessor.transform(df_engineered)
            
            # Prediction
            fraud_prob = model.predict_proba(X_processed)[0][1]
            total_risk = int(fraud_prob * 100)
            
            # SHAP Explanation
            try:
                explainer = ModelExplainer(model, X_processed)
                
                X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.feature_names_out_)
                contributions, base_value = explainer.get_explanation(X_processed_df)
                
                top_contributions = contributions[:4] if contributions else []
                
            except Exception as e:
                top_contributions = []
                st.warning(f"Could not generate SHAP values: {e}")

        st.markdown("<hr style='border-color: var(--border); margin: 2rem 0;'>", unsafe_allow_html=True)
        
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
            
            shap_html = f"""<div style="background: var(--background); padding: 20px; border-radius: 8px; border: 1px solid var(--border); height: 100%;">"""
            
            if not top_contributions:
                shap_html += "<div>No SHAP explanations available.</div>"
            else:
                for c in top_contributions:
                    val = c['Contribution']
                    feat = c['Feature'].replace('num__', '').replace('cat__', '')
                    if val > 0:
                        shap_html += f'<div style="margin-bottom: 12px; display: flex; align-items: center;"><div><strong style="color: var(--danger)">+{val:.3f} Risk:</strong> The feature `{feat}` significantly increased the risk score.</div></div>'
                    else:
                        shap_html += f'<div style="margin-bottom: 12px; display: flex; align-items: center;"><div><strong style="color: var(--success)">{val:.3f} Risk:</strong> The feature `{feat}` decreased the risk score.</div></div>'
                
            shap_html += "</div>"
            st.markdown(shap_html, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid var(--warning); padding: 15px 24px;">
            <div class="metric-title" style="margin-bottom: 5px;">Estimated Financial Exposure</div>
            <div class="metric-value" style="color: var(--warning); font-size: 1.5rem;">${exposure:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

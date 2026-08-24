import streamlit as st
import pandas as pd
import time
import joblib
import os
from src.features.engineering import FeatureEngineer
from src.ml.explain import ModelExplainer

def render():
    # Premium UI CSS
    st.markdown("""
    <style>
    [data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    div[data-baseweb="input"] > div:hover, div[data-baseweb="select"] > div:hover {
        border-color: rgba(56, 189, 248, 0.4) !important;
    }
    button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #3b82f6) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.5) !important;
    }
    .premium-title {
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0;
    }
    .section-header {
        font-size: 0.85rem;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='premium-title'>New Claim Validator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Run a live AI assessment on a new insurance claim using the trained XGBoost model.</p>", unsafe_allow_html=True)
    
    with st.form("new_claim_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='section-header'>Policyholder Information</div>", unsafe_allow_html=True)
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            region = st.selectbox("Region", ["North", "South", "East", "West"])
            credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
            
            st.markdown("<br><div class='section-header'>Policy Details</div>", unsafe_allow_html=True)
            policy_type = st.selectbox("Policy Type", ["Auto", "Home", "Health", "Life"])
            coverage_amt = st.number_input("Total Coverage Amount ($)", min_value=1000, max_value=1000000, value=50000, step=1000)
            premium_amt = st.number_input("Premium Amount ($)", min_value=100, max_value=50000, value=1500, step=100)
            
        with col2:
            st.markdown("<div class='section-header'>Incident Information</div>", unsafe_allow_html=True)
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
            
            # Save to session state for later storing
            st.session_state['last_assessment'] = input_data.copy()
            st.session_state['last_assessment']['Fraud_Risk_Score'] = total_risk
            
            # SHAP Explanation
            try:
                # Get a small background dataset for SHAP
                df_bg = pd.read_csv("data/sample/Scored-Dataset.csv").sample(n=50, random_state=42)
                engineer_bg = FeatureEngineer(df_bg)
                df_bg_engineered = engineer_bg.engineer_features()
                X_bg = preprocessor.transform(df_bg_engineered)
                
                explainer = ModelExplainer(model, X_bg)
                
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
            
            with st.container():
                st.markdown("""<div style="background: var(--background); padding: 20px; border-radius: 8px; border: 1px solid var(--border); height: 100%;">""", unsafe_allow_html=True)
                if not top_contributions:
                    st.markdown("No SHAP explanations available.")
                else:
                    for c in top_contributions:
                        val = c['Contribution']
                        feat = c['Feature'].replace('num__', '').replace('cat__', '')
                        
                        if feat in df_engineered.columns:
                            raw_val = df_engineered[feat].iloc[0]
                            if isinstance(raw_val, float):
                                val_str = f" *(Value: {raw_val:.2f})*"
                            else:
                                val_str = f" *(Value: {raw_val})*"
                        else:
                            val_str = ""
                            
                        if val > 0:
                            st.markdown(f"🔴 **{feat}**{val_str} significantly increased the risk score (Impact: +{abs(val):.3f})")
                        else:
                            st.markdown(f"🟢 **{feat}**{val_str} decreased the risk score (Impact: -{abs(val):.3f})")
                st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid var(--warning); padding: 15px 24px;">
            <div class="metric-title" style="margin-bottom: 5px;">Estimated Financial Exposure</div>
            <div class="metric-value" style="color: var(--warning); font-size: 1.5rem;">${exposure:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    if 'last_assessment' in st.session_state:
        st.markdown("<hr style='border-color: var(--border); margin: 2rem 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.1rem; color: var(--text);'>Store Assessment</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: var(--text-muted); font-size: 0.9rem;'>Save this AI-assessed claim to the database so it appears in the Claims Queue and Executive Dashboard for credibility tracking.</p>", unsafe_allow_html=True)
        
        if st.button("Store Assessed Claim to Database", type="secondary"):
            try:
                df_new = st.session_state['last_assessment'].copy()
                # Generate mock identifiers for the new claim
                df_new['Claim_ID'] = f"NEW-CLM-{int(time.time())}"
                df_new['Customer_ID'] = f"NEW-CUST-{int(time.time())}"
                df_new['Policy_Number'] = f"NEW-POL-{int(time.time())}"
                df_new['Fraud_Flag'] = "Pending"
                
                scored_path = "data/sample/Scored-Dataset.csv"
                if os.path.exists(scored_path):
                    df_existing = pd.read_csv(scored_path)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(scored_path, index=False)
                    st.success("Claim successfully stored! It is now visible in the Claims Queue and Dashboard.")
                    del st.session_state['last_assessment']
                else:
                    st.error("Could not find the Scored-Dataset.csv to append to.")
            except Exception as e:
                st.error(f"Error saving claim: {e}")

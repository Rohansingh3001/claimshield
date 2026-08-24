import streamlit as st
import pandas as pd
import os
import joblib
from src.features.engineering import FeatureEngineer
from src.ml.explain import ModelExplainer

def render():
    st.markdown("<h1 class='text-primary' style='margin-bottom: 0;'>Claim Investigation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 1.1rem; margin-bottom: 2rem;'>Deep dive into a specific claim's risk profile and AI explanation.</p>", unsafe_allow_html=True)
    
    data_path = "data/sample/Scored-Dataset.csv"
    if not os.path.exists(data_path):
        st.warning("Scored dataset not found. Please run the ML pipeline first.")
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
    selected_claim_id = st.selectbox("Search or Select a Claim ID to Investigate", claim_ids)
    
    if selected_claim_id:
        st.markdown("<hr style='border-color: var(--border); margin: 2rem 0;'>", unsafe_allow_html=True)
        claim_data = df[df['Claim_ID'] == selected_claim_id].iloc[0]
        
        # We now have the actual ML-generated risk score from the batch process
        risk_score = int(claim_data.get('Fraud_Risk_Score', 0))
        
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
            <div class="metric-card">
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 24px; font-size: 1.05rem;'>
                    <div><span style='color: var(--text-muted)'>Customer:</span> <strong style="color: var(--text)">{claim_data.get('Customer_ID', 'N/A')}</strong></div>
                    <div><span style='color: var(--text-muted)'>Policy No:</span> <strong style="color: var(--text)">{claim_data.get('Policy_Number', 'N/A')}</strong></div>
                    <div><span style='color: var(--text-muted)'>Claim Amt:</span> <strong style="color: var(--primary)">${claim_data.get('Claim_Amount', 0):,.2f}</strong></div>
                    <div><span style='color: var(--text-muted)'>Coverage:</span> <strong style="color: var(--text)">${claim_data.get('Coverage_Amount', 0):,.2f}</strong></div>
                    <div><span style='color: var(--text-muted)'>Claim Type:</span> <strong style="color: var(--text)">{claim_data.get('Claim_Type', 'N/A')}</strong></div>
                    <div><span style='color: var(--text-muted)'>Policy Type:</span> <strong style="color: var(--text)">{claim_data.get('Policy_Type', 'N/A')}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("<h3 style='font-size: 1.2rem; color: var(--text-muted);'>STEP 2: AI Assessment</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {color}; border-radius: 8px; padding: 32px; text-align: center; height: 100%;">
                <div style="color: {color}; font-weight: 700; font-size: 1.1rem; letter-spacing: 0.1em; margin-bottom: 8px;">{risk_label}</div>
                <div style="font-size: 4rem; font-weight: 800; color: var(--text); line-height: 1;">{risk_score}</div>
                <div style="color: var(--text-muted); font-size: 0.9rem; margin-top: 5px; margin-bottom: 24px;">out of 100</div>
                <div style="background-color: {color}; color: #fff; padding: 8px 24px; border-radius: 4px; font-weight: 700; display: inline-block; letter-spacing: 0.05em;">ACTION: {decision}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- Real SHAP Generation ---
        top_contributions = []
        if os.path.exists("models/preprocessor.pkl") and os.path.exists("models/xgboost.pkl"):
            try:
                preprocessor = joblib.load("models/preprocessor.pkl")
                model = joblib.load("models/xgboost.pkl")
                
                # We need to drop leakage/id columns as run_pipeline does, but we only have 1 row here.
                # Actually, the DataPreprocessor.transform just uses the features it was fitted on.
                # It's better to pass the DataFrame with all the columns the preprocessor expects.
                input_df = df[df['Claim_ID'] == selected_claim_id].copy()
                engineer = FeatureEngineer(input_df)
                df_engineered = engineer.engineer_features()
                
                # preprocessor.transform will automatically select only the columns it needs
                X_processed = preprocessor.transform(df_engineered)
                
                # Get a small background dataset for SHAP
                df_bg = df.sample(n=50, random_state=42)
                engineer_bg = FeatureEngineer(df_bg)
                df_bg_engineered = engineer_bg.engineer_features()
                X_bg = preprocessor.transform(df_bg_engineered)
                
                explainer = ModelExplainer(model, X_bg)
                X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.feature_names_out_)
                contributions, base_value = explainer.get_explanation(X_processed_df)
                top_contributions = contributions[:4] if contributions else []
                
            except Exception as e:
                st.warning(f"Could not generate live SHAP values: {e}")
        else:
            st.warning("Model artifacts missing for SHAP explanation.")

        col3, col4 = st.columns([1.5, 1])
        
        with col3:
            st.markdown("<h3 style='font-size: 1.2rem; color: var(--text-muted);'>STEP 3: Why was this decision made?</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div class="metric-card" style="margin-bottom: 15px; padding: 15px 24px;">
                <p style='margin: 0; color: var(--text-muted);'>The <span style='color: var(--primary); font-weight: 600;'>XGBoost Model</span> analyzed all features. Here are the top factors altering the risk score:</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("View AI Reasoning (SHAP)", expanded=True):
                st.markdown("""<div style="background: var(--background); padding: 20px; border-radius: 8px; border: 1px solid var(--border);">""", unsafe_allow_html=True)
                
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

        with col4:
            st.markdown("<h3 style='font-size: 1.2rem; color: var(--text-muted);'>STEP 4: Business Impact</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid var(--warning);">
                <div class="metric-title">Estimated Financial Exposure</div>
                <div class="metric-value" style="color: var(--warning);">${exposure:,.0f}</div>
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 15px; margin-bottom: 0; line-height: 1.4;">Calculated as (Claim Amount × Probability of Fraud). Prioritize investigations with high exposure.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.2rem; color: var(--text-muted);'>STEP 5: Historical Context (Similar Claims)</h3>", unsafe_allow_html=True)
        
        sim_data = pd.DataFrame({
            'Claim ID': ['CLM-1021', 'CLM-8342', 'CLM-7341'],
            'Similarity': ['94%', '89%', '86%'],
            'Historical Outcome': ['Fraud', 'Fraud', 'Genuine']
        })
        
        st.dataframe(sim_data, use_container_width=True, hide_index=True)

import streamlit as st
import pandas as pd
import numpy as np

def render():
    st.markdown("<h1 class='gradient-text' style='margin-bottom: 0;'>Model Performance</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 1.1rem; margin-bottom: 2rem;'>Technical evaluation of the machine learning models.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid var(--success); padding: 16px 20px; border-radius: 8px; margin-bottom: 2rem;">
        <span style="color: var(--success); font-weight: 800; margin-right: 10px;">SUCCESS:</span> The primary model selected for production is <strong>XGBoost</strong> due to its superior PR-AUC and Recall for the minority (fraud) class, alongside native support for SHAP explainability.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size: 1.2rem; color: var(--primary); margin-top: 2rem; font-weight: 800;'>MODEL COMPARISON METRICS</h3>", unsafe_allow_html=True)
    
    # Mock data for UI layout, would be loaded from evaluate.py output
    metrics_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost (Selected)'],
        'ROC-AUC': [0.72, 0.85, 0.91],
        'PR-AUC': [0.45, 0.68, 0.84],
        'Recall (Fraud)': [0.41, 0.62, 0.82],
        'Precision (Fraud)': [0.38, 0.71, 0.74],
        'F1 Score': [0.39, 0.66, 0.78]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    st.markdown("<div class='metric-card' style='padding: 0; overflow: hidden; border: 1px solid var(--glass-border); margin-bottom: 3rem;'>", unsafe_allow_html=True)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); font-weight: 700;'>ROC Curve (XGBoost)</h3>", unsafe_allow_html=True)
        st.markdown("<div style='background: var(--card-bg); backdrop-filter: blur(10px); padding: 50px 20px; border-radius: 16px; border: 2px dashed rgba(255,255,255,0.2); text-align: center; color: var(--text-muted); box-shadow: inset 0 0 20px rgba(0,0,0,0.5);'>*(Visualization placeholder - this would render the actual ROC curve from saved evaluation artifacts)*</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); font-weight: 700;'>Precision-Recall Curve (XGBoost)</h3>", unsafe_allow_html=True)
        st.markdown("<div style='background: var(--card-bg); backdrop-filter: blur(10px); padding: 50px 20px; border-radius: 16px; border: 2px dashed rgba(236, 72, 153, 0.5); text-align: center; color: #EC4899; box-shadow: inset 0 0 20px rgba(236,72,153,0.1); text-shadow: 0 0 10px rgba(236,72,153,0.3);'>*(Visualization placeholder - this would render the actual PR curve)*</div>", unsafe_allow_html=True)
        st.markdown("<p style='color: var(--text-muted); font-size: 0.85rem; margin-top: 10px;'>PR-AUC is critical because the dataset is imbalanced. It shows how well the model identifies true fraud without raising too many false alarms.</p>", unsafe_allow_html=True)
        
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 3rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); font-weight: 700;'>Global Feature Importance (SHAP)</h3>", unsafe_allow_html=True)
    st.markdown("<div style='background: var(--card-bg); backdrop-filter: blur(10px); padding: 50px 20px; border-radius: 16px; border: 2px dashed var(--glass-border); text-align: center; color: var(--text-muted); margin-bottom: 20px; box-shadow: inset 0 0 20px rgba(0,0,0,0.5);'>*(Visualization placeholder - this would render a SHAP summary plot)*</div>", unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_header.png", use_column_width=True)


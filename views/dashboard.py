import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

def render():
    st.markdown("<h1 class='gradient-text' style='margin-bottom: 0;'>Executive Dashboard & Model Performance</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 1.1rem; margin-bottom: 2rem;'>Overview of claims, fraud detection metrics, and financial exposure.</p>", unsafe_allow_html=True)
    
    # Load data for dashboard
    data_path = "data/sample/Scored-Dataset.csv"
    if not os.path.exists(data_path):
        st.warning("Scored dataset not found. Please run the ML pipeline first.")
        return
        
    @st.cache_data
    def load_data():
        df = pd.read_csv(data_path)
        return df
        
    df = load_data()
    
    # If for some reason Fraud_Risk_Score isn't there, error out
    if 'Fraud_Risk_Score' not in df.columns:
        st.error("Fraud_Risk_Score missing. Re-run ML Pipeline.")
        return
    
    total_claims = len(df)
    fraudulent_claims = len(df[df['Fraud_Flag'].astype(str).str.lower().isin(['yes', '1', 'true', 'fraud'])])
    high_risk_claims = len(df[df['Fraud_Risk_Score'] > 70])
    
    # Estimate exposure
    if 'Claim_Amount' in df.columns:
        exposure = df[df['Fraud_Risk_Score'] > 70]['Claim_Amount'].sum()
        exposure_str = f"₹{exposure:,.0f}"
    else:
        exposure_str = "N/A"
        
    fraud_rate = (fraudulent_claims / total_claims) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title" title="Total number of claims processed by the system.">Total Claims</div>
            <div class="metric-value">{total_claims:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid var(--danger);">
            <div class="metric-title" title="Claims with a Fraud Risk Score above 70/100.">High Risk Claims</div>
            <div class="metric-value" style="color: var(--danger);">{high_risk_claims:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid var(--warning);">
            <div class="metric-title" title="Sum of Claim Amounts for all High Risk claims.">Est. Exposure</div>
            <div class="metric-value" style="color: var(--warning);">{exposure_str}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title" title="Percentage of historical claims flagged as fraudulent.">Historical Fraud Rate</div>
            <div class="metric-value">{fraud_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Plotly corporate dark theme layout base
    dark_layout = dict(
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#94A3B8', family='Inter'),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor='#334155', zerolinecolor='#334155'),
        yaxis=dict(gridcolor='#334155', zerolinecolor='#334155')
    )
    
    with col1:
        st.markdown("<h3 style='font-size: 1.1rem; color: var(--text);'>Risk Score Distribution</h3>", unsafe_allow_html=True)
        fig1 = px.histogram(df, x='Fraud_Risk_Score', nbins=20, 
                           color_discrete_sequence=['#3B82F6'])
        
        # Add border to bars for crisp look
        fig1.update_traces(marker=dict(line=dict(width=1, color='#1E293B')))
        
        fig1.update_layout(**dark_layout, showlegend=False, bargap=0.1)
        # Add risk threshold lines
        fig1.add_vline(x=30, line_dash="dash", line_color="#10B981", annotation_text="Low", annotation_position="top left")
        fig1.add_vline(x=70, line_dash="dash", line_color="#EF4444", annotation_text="High", annotation_position="top right")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.markdown("<h3 style='font-size: 1.1rem; color: var(--text);'>Financial Impact Matrix</h3>", unsafe_allow_html=True)
        if 'Claim_Amount' in df.columns:
            # Sample for performance
            sample_df = df.sample(min(1500, len(df)))
            fig2 = px.scatter(sample_df, x='Claim_Amount', y='Fraud_Risk_Score', 
                             color='Fraud_Risk_Score', 
                             color_continuous_scale=['#10B981', '#3B82F6', '#1E40AF', '#EF4444'],
                             opacity=0.8)
            
            fig2.update_traces(marker=dict(size=7, line=dict(width=0.5, color='#0F172A')))
            fig2.update_layout(**dark_layout, coloraxis_showscale=False)
            fig2.add_hline(y=70, line_dash="dot", line_color="#EF4444")
            st.plotly_chart(fig2, use_container_width=True)

    # --- Model Performance Section ---
    st.markdown("<hr style='border-color: var(--border); margin: 3rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h2 class='text-primary' style='margin-bottom: 0;'>Model Performance</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 1.1rem; margin-bottom: 2rem;'>Technical evaluation of the machine learning models.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid var(--success); padding: 16px 20px; border-radius: 4px; margin-bottom: 2rem;">
        <span style="color: var(--success); font-weight: 700; margin-right: 10px;">SUCCESS:</span> The primary model selected for production is <strong>XGBoost</strong> due to its superior PR-AUC and Recall for the minority (fraud) class, alongside native support for SHAP explainability.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); margin-top: 2rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;'>Model Evaluation Metrics (XGBoost)</h3>", unsafe_allow_html=True)
    
    metrics_path = "models/metrics.json"
    comparison_path = "models/model_comparison.csv"
    
    if os.path.exists(comparison_path):
        df_comparison = pd.read_csv(comparison_path)
        st.markdown("<h3 style='font-size: 1.1rem; color: var(--text);'>Model Comparison</h3>", unsafe_allow_html=True)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    elif os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            real_metrics = json.load(f)
        
        metrics_data = {
            'Model': ['XGBoost (Selected)'],
            'Accuracy': [round(real_metrics.get('Accuracy', 0), 4)],
            'ROC-AUC': [round(real_metrics.get('ROC-AUC', 0), 4)],
            'PR-AUC': [round(real_metrics.get('PR-AUC', 0), 4)],
            'Recall': [round(real_metrics.get('Recall', 0), 4)],
            'Precision': [round(real_metrics.get('Precision', 0), 4)],
            'F1 Score': [round(real_metrics.get('F1 Score', 0), 4)]
        }
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
    # Confusion Matrix
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            rm = json.load(f)
            tn = rm.get('TN', 0)
            fp = rm.get('FP', 0)
            fn = rm.get('FN', 0)
            tp = rm.get('TP', 0)
            
            st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); margin-top: 2rem;'>Confusion Matrix (XGBoost)</h3>", unsafe_allow_html=True)
            
            cm_html = f"""
            <table style="width: 100%; max-width: 400px; border-collapse: collapse; text-align: center; color: var(--text);">
                <tr>
                    <td style="border: none;"></td>
                    <td style="border: 1px solid var(--border); padding: 10px; background: rgba(255,255,255,0.05);">Predicted Normal</td>
                    <td style="border: 1px solid var(--border); padding: 10px; background: rgba(255,255,255,0.05);">Predicted Fraud</td>
                </tr>
                <tr>
                    <td style="border: 1px solid var(--border); padding: 10px; background: rgba(255,255,255,0.05);">Actual Normal</td>
                    <td style="border: 1px solid var(--border); padding: 15px; font-weight: 700; background: rgba(16, 185, 129, 0.1);">{tn}<br><span style="font-size: 0.8em; font-weight: 400;">(True Negative)</span></td>
                    <td style="border: 1px solid var(--border); padding: 15px; font-weight: 700; background: rgba(239, 68, 68, 0.1);">{fp}<br><span style="font-size: 0.8em; font-weight: 400;">(False Positive)</span></td>
                </tr>
                <tr>
                    <td style="border: 1px solid var(--border); padding: 10px; background: rgba(255,255,255,0.05);">Actual Fraud</td>
                    <td style="border: 1px solid var(--border); padding: 15px; font-weight: 700; background: rgba(239, 68, 68, 0.1);">{fn}<br><span style="font-size: 0.8em; font-weight: 400;">(False Negative)</span></td>
                    <td style="border: 1px solid var(--border); padding: 15px; font-weight: 700; background: rgba(16, 185, 129, 0.1);">{tp}<br><span style="font-size: 0.8em; font-weight: 400;">(True Positive)</span></td>
                </tr>
            </table>
            """
            st.markdown(cm_html, unsafe_allow_html=True)
    
    perf_col1, perf_col2 = st.columns(2)
    
    curves_path = "models/curves.json"
    if os.path.exists(curves_path):
        with open(curves_path, 'r') as f:
            curves = json.load(f)
            
        with perf_col1:
            st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); font-weight: 600;'>ROC Curve (XGBoost)</h3>", unsafe_allow_html=True)
            roc_fig = px.line(x=curves['roc']['fpr'], y=curves['roc']['tpr'], labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
            roc_fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            roc_fig.update_layout(**dark_layout)
            st.plotly_chart(roc_fig, use_container_width=True)
            
        with perf_col2:
            st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); font-weight: 600;'>Precision-Recall Curve (XGBoost)</h3>", unsafe_allow_html=True)
            pr_fig = px.line(x=curves['pr']['recall'], y=curves['pr']['precision'], labels={'x': 'Recall', 'y': 'Precision'})
            pr_fig.update_layout(**dark_layout)
            st.plotly_chart(pr_fig, use_container_width=True)
            st.markdown("<p style='color: var(--text-muted); font-size: 0.85rem; margin-top: 10px;'>PR-AUC is critical because the dataset is imbalanced. It shows how well the model identifies true fraud without raising too many false alarms.</p>", unsafe_allow_html=True)
    else:
        with perf_col1:
            st.markdown("*(Run ML pipeline to view ROC Curve)*")
        with perf_col2:
            st.markdown("*(Run ML pipeline to view PR Curve)*")
        
    st.markdown("<hr style='border-color: var(--border); margin: 3rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); font-weight: 600;'>Global Feature Importance</h3>", unsafe_allow_html=True)
    
    import joblib
    if os.path.exists("models/xgboost.pkl") and os.path.exists("models/preprocessor.pkl"):
        model = joblib.load("models/xgboost.pkl")
        preprocessor = joblib.load("models/preprocessor.pkl")
        
        # XGBoost CalibratedClassifierCV wraps the actual model in calibrated_classifiers_
        # We can extract the underlying feature importances from the first calibrated model
        try:
            base_estimator = model.calibrated_classifiers_[0].estimator
            importances = base_estimator.feature_importances_
            feature_names = preprocessor.feature_names_out_
            
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            imp_df = imp_df.sort_values(by='Importance', ascending=False).head(10)
            imp_df['Feature'] = imp_df['Feature'].str.replace('num__', '').str.replace('cat__', '')
            
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h')
            fig_imp.update_layout(**dark_layout, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not extract feature importances: {e}")
            
    # Validation Table
    st.markdown("<hr style='border-color: var(--border); margin: 3rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); font-weight: 600;'>Test Set Validation (Ground Truth vs Prediction)</h3>", unsafe_allow_html=True)
    
    val_path = "data/sample/Test-Validation-Results.csv"
    if os.path.exists(val_path):
        val_df = pd.read_csv(val_path)
        st.dataframe(val_df.head(100), use_container_width=True)
        st.markdown("<p style='color: var(--text-muted); font-size: 0.85rem;'>Showing top 100 predictions from the unseen 20% test set.</p>", unsafe_allow_html=True)
    else:
        st.warning("Run ML Pipeline to generate Test-Validation-Results.csv")


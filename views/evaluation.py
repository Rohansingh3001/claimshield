import streamlit as st
import pandas as pd
import os
import joblib
from src.features.engineering import FeatureEngineer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

def render():
    st.markdown("<h1 class='text-primary' style='margin-bottom: 0;'>Batch Evaluation & Testing</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 1.1rem; margin-bottom: 2rem;'>Upload an unseen dataset to evaluate the AI model without retraining.</p>", unsafe_allow_html=True)
    
    if not os.path.exists("models/preprocessor.pkl") or not os.path.exists("models/xgboost.pkl"):
        st.error("Model artifacts not found! Please run the ML pipeline first.")
        return
        
    uploaded_file = st.file_uploader("Upload External Dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            df_new = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded dataset with {len(df_new):,} records.")
            
            with st.spinner("Running AI Assessment..."):
                preprocessor = joblib.load("models/preprocessor.pkl")
                model = joblib.load("models/xgboost.pkl")
                
                # 1. Feature Engineering
                engineer = FeatureEngineer(df_new)
                df_engineered = engineer.engineer_features()
                
                # 2. Preprocessing (Transform ONLY)
                X_processed = preprocessor.transform(df_engineered)
                
                # 3. Prediction
                fraud_probs = model.predict_proba(X_processed)[:, 1]
                threshold = 0.50
                predictions = (fraud_probs >= threshold).astype(int)
                
                df_results = df_new.copy()
                df_results['Fraud_Risk_Score'] = (fraud_probs * 100).round(2)
                df_results['Predicted_Label'] = ['Fraud' if p == 1 else 'Normal' for p in predictions]
                
                # 4. Ground Truth Comparison (if Fraud_Flag exists)
                if 'Fraud_Flag' in df_results.columns:
                    st.markdown("<hr style='border-color: var(--border); margin: 2rem 0;'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); font-weight: 600;'>Evaluation Metrics</h3>", unsafe_allow_html=True)
                    
                    # Convert to binary
                    y_true = df_results['Fraud_Flag'].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true', 'fraud'] else 0)
                    
                    acc = accuracy_score(y_true, predictions)
                    rec = recall_score(y_true, predictions)
                    prec = precision_score(y_true, predictions, zero_division=0)
                    f1 = f1_score(y_true, predictions)
                    roc = roc_auc_score(y_true, fraud_probs)
                    pr = average_precision_score(y_true, fraud_probs)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Recall (Fraud Detected)", f"{rec*100:.1f}%")
                    col2.metric("Precision", f"{prec*100:.1f}%")
                    col3.metric("F1 Score", f"{f1*100:.1f}%")
                    col4.metric("Accuracy", f"{acc*100:.1f}%")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="background-color: rgba(30, 41, 59, 0.5); border-left: 4px solid var(--primary); padding: 20px; border-radius: 4px; margin-bottom: 2rem;">
                        <h4 style="color: var(--primary); margin-top: 0; margin-bottom: 10px; font-size: 1rem;">Why this Threshold (0.50)? — The Business Trade-off</h4>
                        <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.5; margin-bottom: 0;">
                            In fraud detection, there is a fundamental trade-off between <strong>Recall</strong> (catching as much fraud as possible) and <strong>Precision</strong> (ensuring every flagged claim is actually fraud). 
                            <br><br>
                            We use a classification threshold of <strong>0.50</strong>. This balances <strong>Recall</strong> and <strong>Precision</strong>, providing a balanced approach to flagging claims. The business logic dictates that the administrative cost of a manual investigator reviewing a "false alarm" is significantly cheaper than the direct financial loss of paying out a fraudulent $80,000 claim that slipped through the cracks.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_true, predictions)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        
                        st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); margin-top: 2rem;'>Confusion Matrix</h3>", unsafe_allow_html=True)
                        cm_html = f"""
                        <table style="width: 100%; max-width: 400px; border-collapse: collapse; text-align: center; color: var(--text);">
                            <tr>
                                <td style="border: none;"></td>
                                <td style="border: 1px solid var(--border); padding: 10px; background: rgba(255,255,255,0.05);">Predicted Normal</td>
                                <td style="border: 1px solid var(--border); padding: 10px; background: rgba(255,255,255,0.05);">Predicted Fraud</td>
                            </tr>
                            <tr>
                                <td style="border: 1px solid var(--border); padding: 10px; background: rgba(255,255,255,0.05);">Actual Normal</td>
                                <td style="border: 1px solid var(--border); padding: 15px; font-weight: 700; background: rgba(16, 185, 129, 0.1);">{tn}</td>
                                <td style="border: 1px solid var(--border); padding: 15px; font-weight: 700; background: rgba(239, 68, 68, 0.1);">{fp}</td>
                            </tr>
                            <tr>
                                <td style="border: 1px solid var(--border); padding: 10px; background: rgba(255,255,255,0.05);">Actual Fraud</td>
                                <td style="border: 1px solid var(--border); padding: 15px; font-weight: 700; background: rgba(239, 68, 68, 0.1);">{fn}</td>
                                <td style="border: 1px solid var(--border); padding: 15px; font-weight: 700; background: rgba(16, 185, 129, 0.1);">{tp}</td>
                            </tr>
                        </table>
                        """
                        st.markdown(cm_html, unsafe_allow_html=True)
                        
                    # Validation Table
                    df_results['Actual_Label'] = ['Fraud' if y == 1 else 'Normal' for y in y_true]
                    df_results['Result'] = ['Correct' if a == p else 'Incorrect' for a, p in zip(df_results['Actual_Label'], df_results['Predicted_Label'])]
                    
                    st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); margin-top: 2rem;'>Validation Results</h3>", unsafe_allow_html=True)
                    st.dataframe(df_results[['Claim_ID', 'Actual_Label', 'Predicted_Label', 'Fraud_Risk_Score', 'Result']], use_container_width=True)
                    
                else:
                    st.markdown("<hr style='border-color: var(--border); margin: 2rem 0;'>", unsafe_allow_html=True)
                    st.info("No 'Fraud_Flag' column found in the uploaded dataset. Inference completed successfully without evaluation metrics.")
                    st.markdown("<h3 style='font-size: 1.1rem; color: var(--text); margin-top: 1rem;'>Prediction Results</h3>", unsafe_allow_html=True)
                    display_cols = ['Claim_ID', 'Fraud_Risk_Score', 'Predicted_Label']
                    display_cols = [c for c in display_cols if c in df_results.columns]
                    st.dataframe(df_results[display_cols], use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error processing dataset: {e}")

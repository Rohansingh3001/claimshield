# ClaimShield AI

Insurance Fraud Detection & Investigation Platform.

## Overview

ClaimShield AI is an investigator decision-support tool designed to identify potentially fraudulent claims, assign a Fraud Risk Score, explain the factors contributing to the prediction (using SHAP), estimate potential financial exposure, and recommend an appropriate investigation action.

## Tech Stack
- **Data & ML**: pandas, scikit-learn, XGBoost, SHAP
- **Web App**: Streamlit, Plotly, Custom CSS

## Project Structure
```text
claimshield/
├── app.py (Main Streamlit Entry)
├── pages/ (Streamlit UI Pages)
├── src/ (Core Logic)
│   ├── data/ (Loading, Validation, Preprocessing)
│   ├── features/ (Engineering)
│   ├── ml/ (Training, Evaluation, Explainability)
│   ├── risk/ (Scoring & Decision thresholds)
│   └── similarity/ (KNN Historical Similarities)
├── data/
│   └── sample/ (Dataset goes here)
└── models/ (Saved model artifacts)
```

## Setup & Running

1. **Install Requirements**:
   ```bash
   python -m venv venv
   source venv/Scripts/activate # On Windows
   pip install -r requirements.txt
   ```

2. **Run ML Pipeline**:
   Make sure `ClaimShieldAI-Dataset.csv` is in `data/sample/`.
   ```bash
   python run_pipeline.py
   ```

3. **Run Streamlit App**:
   ```bash
   streamlit run app.py
   ```

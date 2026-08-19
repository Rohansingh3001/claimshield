import streamlit as st

st.set_page_config(
    page_title="ClaimShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich, premium aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --primary: #2563EB;
        --secondary: #3B82F6;
        --background: #0F172A;
        --card-bg: #1E293B;
        --text: #F8FAFC;
        --text-muted: #94A3B8;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --border: #334155;
    }
    
    /* Global Background & Font */
    .stApp {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography Gradients (Removed, replaced with solid color) */
    .gradient-text {
        color: var(--text);
        font-weight: 700;
    }
    .text-primary {
        color: var(--primary);
    }
    
    /* Premium Minimalist Card */
    .metric-card {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 20px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid var(--border);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        border-color: #475569;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-title {
        color: var(--text-muted);
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.2;
    }
    
    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0B1120 !important;
        border-right: 1px solid var(--border);
    }
    
    /* Badges */
    .badge {
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .badge-high { background-color: rgba(239, 68, 68, 0.1); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.2); }
    .badge-medium { background-color: rgba(245, 158, 11, 0.1); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.2); }
    .badge-low { background-color: rgba(16, 185, 129, 0.1); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.2); }
</style>

""", unsafe_allow_html=True)

# Sidebar Design
st.sidebar.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 30px;">
    <div style="font-size: 28px; margin-right: 12px; color: #3B82F6;">🛡️</div>
    <div>
        <h2 style="margin: 0; font-size: 20px; font-weight: 700; color: #F8FAFC;">ClaimShield AI</h2>
        <span style="font-size: 11px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em;">Decision Support</span>
    </div>
</div>
""", unsafe_allow_html=True)

pages = {
    "📊 Executive Dashboard": "views.dashboard",
    "✨ New Claim Simulator": "views.predict",
    "🔍 Claim Investigation": "views.investigation",
    "📑 Claims Queue": "views.claims",
    "⚙️ Model Performance": "views.model_performance"
}

selection = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
with st.sidebar.expander("ℹ️ How it Works"):
    st.markdown("""
    **ClaimShield AI** analyzes claims using an XGBoost model.
    - Extracts 30+ features.
    - Calculates a Risk Score (0-100).
    - Uses **SHAP** to explain *why* a claim is suspicious.
    - Suggests an action: Approve, Review, or Investigate.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Production Build v2.0 | High-Contrast Mode")

# Dynamically load the selected page
import importlib
try:
    page_module = importlib.import_module(pages[selection])
    page_module.render()
except Exception as e:
    st.error(f"Error loading page: {e}")

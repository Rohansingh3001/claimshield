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
    :root {
        --primary: #6366F1;
        --primary-dark: #4F46E5;
        --background: #0B0F19;
        --card-bg: #151B2B;
        --text: #F8FAFC;
        --text-muted: #94A3B8;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --border: rgba(255, 255, 255, 0.08);
    }
    
    /* Global Background & Font */
    .stApp {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Card Styles */
    .metric-card {
        background-color: var(--card-bg);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.15);
        border: 1px solid var(--border);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .metric-title {
        color: var(--text-muted);
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
        color: var(--text);
        line-height: 1.2;
    }
    
    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #080C14;
        border-right: 1px solid var(--border);
    }
    
    /* Badges */
    .badge {
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .badge-high { background-color: rgba(239, 68, 68, 0.15); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-medium { background-color: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-low { background-color: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }
</style>
""", unsafe_allow_html=True)

# Sidebar Design
st.sidebar.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 30px;">
    <div style="font-size: 32px; margin-right: 12px;">🛡️</div>
    <div>
        <h2 style="margin: 0; font-size: 20px; font-weight: 700; background: -webkit-linear-gradient(45deg, #6366F1, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">ClaimShield AI</h2>
        <span style="font-size: 11px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em;">Decision Support</span>
    </div>
</div>
""", unsafe_allow_html=True)

pages = {
    "📊 Executive Dashboard": "views.dashboard",
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
st.sidebar.caption("Hackathon MVP v2.0 | High-Contrast Mode")

# Dynamically load the selected page
import importlib
try:
    page_module = importlib.import_module(pages[selection])
    page_module.render()
except Exception as e:
    st.error(f"Error loading page: {e}")

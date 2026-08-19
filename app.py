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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary: #4F46E5;
        --primary-glow: rgba(79, 70, 229, 0.5);
        --secondary: #EC4899;
        --background: #050509;
        --card-bg: rgba(20, 22, 35, 0.6);
        --text: #F8FAFC;
        --text-muted: #94A3B8;
        --success: #10B981;
        --success-glow: rgba(16, 185, 129, 0.4);
        --warning: #F59E0B;
        --warning-glow: rgba(245, 158, 11, 0.4);
        --danger: #EF4444;
        --danger-glow: rgba(239, 68, 68, 0.5);
        --border: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }
    
    /* Animations */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes pulse-danger {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    @keyframes pulse-success {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* Global Background & Font */
    .stApp {
        background: radial-gradient(circle at 15% 50%, rgba(20, 15, 40, 1), #050509 40%);
        background-color: var(--background);
        color: var(--text);
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Typography Gradients */
    .gradient-text {
        background: linear-gradient(90deg, #6366F1, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Premium Glassmorphic Card */
    .metric-card {
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--glass-border);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    /* Card Glow Effect */
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0; left: -100%;
        width: 50%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
        transition: all 0.5s ease;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px -5px rgba(0, 0, 0, 0.5), 0 0 20px var(--primary-glow);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .metric-title {
        color: var(--text-muted);
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 12px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--text);
        line-height: 1.1;
    }
    
    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(8, 12, 20, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--glass-border);
    }
    
    /* Badges */
    .badge {
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .badge-high { background-color: rgba(239, 68, 68, 0.15); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.4); text-shadow: 0 0 8px var(--danger-glow); }
    .badge-medium { background-color: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-low { background-color: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.4); text-shadow: 0 0 8px var(--success-glow); }
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

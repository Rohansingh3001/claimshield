import streamlit as st

models = [
    {
        "name": "Logistic Regression",
        "accent": "#00D9FF",
        "desc": [
            "Linear model ideal for binary classification tasks.",
            "Simple, interpretable and effective baseline for fraud detection."
        ],
        "accuracy": "0.4567",
        "roc_auc": "0.2648",
        "pr_auc": "0.1783",
        "recall": "0.2578",
        "precision": "0.2678",
        "f1": "0.3136"
    },
    {
        "name": "Random Forest",
        "accent": "#F59E0B",
        "desc": [
            "Ensemble model capturing non-linear relationships.",
            "Reduces overfitting and improves robustness."
        ],
        "accuracy": "0.7328",
        "roc_auc": "0.5156",
        "pr_auc": "0.2650",
        "recall": "0.0619",
        "precision": "0.3083",
        "f1": "0.1031"
    },
    {
        "name": "XGBoost",
        "accent": "#A855F7",
        "desc": [
            "Gradient-boosting model for complex interactions.",
            "Best overall balance between accuracy and generalization."
        ],
        "accuracy": "0.6469",
        "roc_auc": "0.5841",
        "pr_auc": "0.3299",
        "recall": "0.4304",
        "precision": "0.3229",
        "f1": "0.3673"
    },
    {
        "name": "SVM",
        "accent": "#14D8C4",
        "desc": [
            "Effective in high-dimensional spaces.",
            "Works well with clear margins between classes."
        ],
        "accuracy": "0.7519",
        "roc_auc": "0.5223",
        "pr_auc": "0.2580",
        "recall": "0.0000",
        "precision": "0.0000",
        "f1": "0.0000"
    }
]

def render_model_card(model):
    accent = model['accent']
    return f"""
    <div class="mc-card" style="--accent: {accent};">
        <div class="mc-icon"></div>
        <div class="mc-name">{model['name']}</div>
        <div class="mc-divider" style="background: {accent}; box-shadow: 0 0 10px {accent};"></div>
        <ul class="mc-desc">
            <li><span class="mc-check" style="color: {accent};">✔</span> {model['desc'][0]}</li>
            <li><span class="mc-check" style="color: {accent};">✔</span> {model['desc'][1]}</li>
        </ul>
    </div>
    """

def render_performance_table(models):
    rows = ""
    for m in models:
        accent = m['accent']
        rows += f"""
        <tr>
            <td style="color: {accent}; font-weight: 600; text-align: left; padding-left: 20px;">{m['name']}</td>
            <td style="color: {accent};">{m['accuracy']}</td>
            <td style="color: {accent};">{m['roc_auc']}</td>
            <td style="color: {accent};">{m['pr_auc']}</td>
            <td style="color: {accent};">{m['recall']}</td>
            <td style="color: {accent};">{m['precision']}</td>
            <td style="color: {accent};">{m['f1']}</td>
        </tr>
        """
        
    return f"""
    <div class="mc-table-container">
        <table class="mc-table">
            <thead>
                <tr>
                    <th style="text-align: left; padding-left: 20px;">📋 Model</th>
                    <th>🎯 Accuracy</th>
                    <th>📈 ROC-AUC</th>
                    <th>📊 PR-AUC</th>
                    <th>🔍 Recall</th>
                    <th>⚡ Precision</th>
                    <th>⭐ F1 Score</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """

def render_conclusion_banner():
    return """
    <div class="mc-banner">
        <div class="mc-banner-left">
            <div class="mc-trophy">🏆</div>
        </div>
        <div class="mc-banner-divider"></div>
        <div class="mc-banner-text">
            <span style="color: #A855F7; font-weight: bold;">XGBoost</span> delivers the best overall performance across key metrics, making it the <span style="color: #14D8C4; font-weight: bold;">most effective model for fraud detection</span>.
        </div>
        <div class="mc-banner-right">
            <div style="font-size: 24px; color: #168BFF;">»</div>
            <div style="font-size: 24px; color: #00D9FF;">🛡️</div>
        </div>
    </div>
    """

def render():
    css = """
    <style>
        /* Base Container */
        .mc-container {
            background-color: #030B1B;
            padding: 40px;
            border-radius: 12px;
            font-family: 'Inter', sans-serif;
            color: #E2E8F0;
        }

        /* Title Area */
        .mc-title-area {
            text-align: left;
            margin-bottom: 40px;
        }
        .mc-title {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0 0 10px 0;
            color: #F8FAFC;
            letter-spacing: 1px;
        }
        .mc-subtitle {
            font-size: 1.1rem;
            color: #94A3B8;
            margin: 0;
        }

        /* Cards Layout */
        .mc-cards-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }

        /* Card Styling */
        .mc-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--accent);
            border-radius: 12px;
            padding: 24px 20px;
            position: relative;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .mc-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 0 25px var(--accent), inset 0 0 20px rgba(0,0,0,0.4);
        }
        .mc-icon {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: rgba(255,255,255,0.05);
            border: 2px solid var(--accent);
            margin-bottom: 16px;
            box-shadow: 0 0 10px var(--accent);
        }
        .mc-name {
            font-size: 1.25rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 12px;
            color: #F8FAFC;
        }
        .mc-divider {
            width: 40px;
            height: 3px;
            border-radius: 2px;
            margin-bottom: 20px;
        }
        .mc-desc {
            list-style: none;
            padding: 0;
            margin: 0;
            text-align: left;
            font-size: 0.85rem;
            color: #CBD5E1;
            line-height: 1.5;
        }
        .mc-desc li {
            margin-bottom: 10px;
            display: flex;
            align-items: flex-start;
        }
        .mc-check {
            margin-right: 8px;
            font-size: 0.9rem;
        }

        /* Table Styling */
        .mc-table-container {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid #168BFF;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 40px;
            box-shadow: 0 0 20px rgba(22, 139, 255, 0.2);
        }
        .mc-table {
            width: 100%;
            border-collapse: collapse;
        }
        .mc-table th, .mc-table td {
            padding: 16px 12px;
            text-align: center;
            border-bottom: 1px solid rgba(22, 139, 255, 0.2);
            border-right: 1px solid rgba(22, 139, 255, 0.1);
        }
        .mc-table th:last-child, .mc-table td:last-child {
            border-right: none;
        }
        .mc-table tbody tr:last-child td {
            border-bottom: none;
        }
        .mc-table th {
            background: rgba(22, 139, 255, 0.1);
            color: #F8FAFC;
            font-weight: 600;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .mc-table td {
            font-size: 1.05rem;
            font-weight: 500;
            background: rgba(2, 8, 23, 0.4);
        }
        .mc-table tbody tr:hover td {
            background: rgba(22, 139, 255, 0.05);
        }

        /* Banner Styling */
        .mc-banner {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid #00D9FF;
            border-radius: 12px;
            padding: 20px 30px;
            display: flex;
            align-items: center;
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.15), inset 0 0 10px rgba(0, 217, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        .mc-banner-left {
            flex-shrink: 0;
            margin-right: 20px;
        }
        .mc-trophy {
            font-size: 2rem;
            text-shadow: 0 0 15px #F59E0B;
        }
        .mc-banner-divider {
            width: 2px;
            height: 50px;
            background: linear-gradient(to bottom, transparent, rgba(0, 217, 255, 0.5), transparent);
            margin-right: 24px;
        }
        .mc-banner-text {
            flex-grow: 1;
            font-size: 1.15rem;
            color: #E2E8F0;
            line-height: 1.5;
        }
        .mc-banner-right {
            display: flex;
            gap: 15px;
            align-items: center;
            margin-left: 20px;
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .mc-cards-grid { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 768px) {
            .mc-cards-grid { grid-template-columns: 1fr; }
            .mc-table { font-size: 0.85rem; }
            .mc-table th, .mc-table td { padding: 10px 6px; }
            .mc-banner { flex-direction: column; text-align: center; gap: 15px; }
            .mc-banner-divider { width: 100%; height: 2px; background: linear-gradient(to right, transparent, rgba(0, 217, 255, 0.5), transparent); margin: 0; }
        }
    </style>
    """

    html = f"""
    <div class="mc-container">
        <div class="mc-title-area">
            <h1 class="mc-title">MACHINE LEARNING MODEL<br>PERFORMANCE COMPARISON</h1>
            <p class="mc-subtitle">Comparing machine learning models to select<br>the most effective one for fraud detection.</p>
        </div>

        <div class="mc-cards-grid">
            {''.join([render_model_card(m) for m in models])}
        </div>

        {render_performance_table(models)}
        
        {render_conclusion_banner()}
    </div>
    """

    st.markdown(css + html, unsafe_allow_html=True)

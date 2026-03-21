import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime, timedelta
from ai_core import DebtCollectionAPI

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AI Debt Collection Agents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.debtcollectionai.com',
        'Report a bug': 'https://github.com/debtcollectionai/issues',
        'About': '## AI Debt Collection Agents v2.0\nPowered by Multi-Agent AI Systems'
    }
)

# =========================================================
# CUSTOM CSS - MODERN UI THEME
# =========================================================
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root Variables ── */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --success: #10b981;
        --success-light: #d1fae5;
        --warning: #f59e0b;
        --warning-light: #fef3c7;
        --danger: #ef4444;
        --danger-light: #fee2e2;
        --info: #3b82f6;
        --info-light: #dbeafe;
        --bg-primary: #0f1117;
        --bg-secondary: #1a1b26;
        --bg-card: #0f1117;
        --text-primary: #ffffff;
        --text-secondary: #0f1117;
        --border: #2d2f3e;
        --gradient-1: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        --gradient-2: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        --gradient-3: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        --gradient-4: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.3);
        --shadow-lg: 0 10px 25px rgba(0,0,0,0.4);
        --shadow-glow: 0 0 20px rgba(99,102,241,0.15);
    }

    /* ── Global Styles ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .main .block-container {
        padding: 1.5rem 2rem 3rem 2rem;
        max-width: 1400px;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }

    /* ── Sidebar Styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13141f 0%, #1a1b2e 100%);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* ── Header Cards ── */
    .page-header {
        background: linear-gradient(135deg, #1e1f2e 0%, #252640 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }

    .page-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--gradient-1);
    }

    .page-header h1 {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }

    .page-header p {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.6;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-glow);
        border-color: var(--primary);
    }

    .metric-card .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.6rem;
        display: block;
    }

    .metric-card .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-secondary);
        margin-bottom: 0.35rem;
    }

    .metric-card .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.2;
    }

    .metric-card .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.4rem;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
    }

    .metric-delta.positive {
        color: var(--success);
        background: rgba(16,185,129,0.1);
    }

    .metric-delta.negative {
        color: var(--danger);
        background: rgba(239,68,68,0.1);
    }

    /* gradient top bar variants */
    .metric-card.purple::before {
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: var(--gradient-1);
    }
    .metric-card.blue::before {
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: var(--gradient-2);
    }
    .metric-card.green::before {
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: var(--gradient-3);
    }
    .metric-card.amber::before {
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: var(--gradient-4);
    }

    /* ── Section Cards ── */
    .section-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
    }

    .section-card h3 {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Status Badges ── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .badge-success {
        background: rgba(16,185,129,0.15);
        color: var(--success);
        border: 1px solid rgba(16,185,129,0.3);
    }

    .badge-danger {
        background: rgba(239,68,68,0.15);
        color: var(--danger);
        border: 1px solid rgba(239,68,68,0.3);
    }

    .badge-warning {
        background: rgba(245,158,11,0.15);
        color: var(--warning);
        border: 1px solid rgba(245,158,11,0.3);
    }

    .badge-info {
        background: rgba(59,130,246,0.15);
        color: var(--info);
        border: 1px solid rgba(59,130,246,0.3);
    }

    .badge-purple {
        background: rgba(99,102,241,0.15);
        color: var(--primary-light);
        border: 1px solid rgba(99,102,241,0.3);
    }

    /* ── Risk Level Indicators ── */
    .risk-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .risk-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.05));
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.3);
    }

    .risk-medium {
        background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(245,158,11,0.05));
        color: #fcd34d;
        border: 1px solid rgba(245,158,11,0.3);
    }

    .risk-low {
        background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(16,185,129,0.05));
        color: #6ee7b7;
        border: 1px solid rgba(16,185,129,0.3);
    }

    /* ── Progress Steps ── */
    .progress-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 1.5rem 0;
        padding: 0 1rem;
    }

    .progress-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        flex: 1;
    }

    .step-circle {
        width: 40px; height: 40px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.9rem;
        transition: all 0.3s;
    }

    .step-circle.active {
        background: var(--gradient-1);
        color: white;
        box-shadow: 0 0 15px rgba(99,102,241,0.4);
    }

    .step-circle.completed {
        background: var(--success);
        color: white;
    }

    .step-circle.pending {
        background: var(--bg-secondary);
        color: var(--text-secondary);
        border: 2px solid var(--border);
    }

    .step-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-align: center;
    }

    .step-connector {
        flex: 1; height: 2px;
        background: var(--border);
        margin: 0 0.5rem;
        margin-bottom: 1.5rem;
    }

    .step-connector.completed { background: var(--success); }

    /* ── Profile Card ── */
    .profile-card {
        background: linear-gradient(135deg, #1e1f2e, #252640);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
    }

    .profile-avatar {
        width: 72px; height: 72px;
        border-radius: 50%;
        background: var(--gradient-1);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem; font-weight: 800; color: white;
        margin: 0 auto 1rem auto;
        box-shadow: 0 0 20px rgba(99,102,241,0.3);
    }

    .profile-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }

    .profile-id {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
    }

    .profile-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .profile-stat {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.75rem;
    }

    .profile-stat-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        font-weight: 600;
    }

    .profile-stat-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 0.15rem;
    }

    /* ── Analysis Result Panel ── */
    .analysis-panel {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 0.75rem;
    }

    .analysis-panel h4 {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    /* ── Training Animation ── */
    .training-progress {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
    }

    .training-progress h3 {
        color: var(--primary-light);
        margin-bottom: 0.5rem;
    }

    /* ── Data table enhancement ── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: var(--bg-secondary);
        padding: 0.3rem;
        border-radius: 12px;
        border: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.5rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: var(--gradient-1) !important;
        color: white !important;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: var(--gradient-1);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(99,102,241,0.3);
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.4);
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 10px;
    }

    /* ── Toast / Alerts ── */
    .custom-alert {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        font-size: 0.88rem;
        font-weight: 500;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        margin-bottom: 0.75rem;
    }

    .alert-info {
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.25);
        color: #93c5fd;
    }

    .alert-success {
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.25);
        color: #6ee7b7;
    }

    .alert-warning {
        background: rgba(245,158,11,0.1);
        border: 1px solid rgba(245,158,11,0.25);
        color: #fcd34d;
    }

    .alert-danger {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.25);
        color: #fca5a5;
    }

    /* ── Compliance Violation Card ── */
    .violation-card {
        background: rgba(239,68,68,0.06);
        border: 1px solid rgba(239,68,68,0.2);
        border-left: 4px solid var(--danger);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.6rem;
    }

    .violation-card .violation-rule {
        font-weight: 700;
        color: #fca5a5;
        font-size: 0.88rem;
    }

    .violation-card .violation-desc {
        color: var(--text-secondary);
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }

    .warning-card {
        background: rgba(245,158,11,0.06);
        border: 1px solid rgba(245,158,11,0.2);
        border-left: 4px solid var(--warning);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.6rem;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: var(--text-secondary);
        font-size: 0.75rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }

    /* ── Responsive Adjustments ── */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        .page-header {
            padding: 1.5rem;
        }
        .page-header h1 {
            font-size: 1.4rem;
        }
        .metric-card .metric-value {
            font-size: 1.3rem;
        }
        .profile-stats {
            grid-template-columns: 1fr;
        }
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Smooth animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: fadeInUp 0.5s ease-out;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .pulse { animation: pulse 2s infinite; }

    /* ── Gauge styling ── */
    .gauge-container {
        text-align: center;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def render_page_header(icon: str, title: str, description: str):
    st.markdown(f"""
    <div class="page-header animate-in">
        <h1>{icon} {title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(icon: str, label: str, value: str, delta: str = None, 
                       delta_positive: bool = True, color_class: str = "purple"):
    delta_html = ""
    if delta:
        dc = "positive" if delta_positive else "negative"
        arrow = "↑" if delta_positive else "↓"
        delta_html = f'<div class="metric-delta {dc}">{arrow} {delta}</div>'

    st.markdown(f"""
    <div class="metric-card {color_class} animate-in">
        <span class="metric-icon">{icon}</span>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_badge(text: str, badge_type: str = "info"):
    return f'<span class="badge badge-{badge_type}">{text}</span>'


def render_risk_indicator(level: str):
    level_lower = level.lower()
    cls = "risk-low" if level_lower == "low" else ("risk-medium" if level_lower == "medium" else "risk-high")
    icon = "🟢" if level_lower == "low" else ("🟡" if level_lower == "medium" else "🔴")
    return f'<div class="risk-indicator {cls}">{icon} {level.upper()} RISK</div>'


def create_gauge_chart(value: float, title: str, max_val: float = 100):
    """Create a modern gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'color': '#94a3b8'}},
        number={'font': {'size': 32, 'color': '#e2e8f0'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#4a4b5e',
                     'tickfont': {'color': '#94a3b8'}},
            'bar': {'color': '#6366f1'},
            'bgcolor': '#1e1f2e',
            'borderwidth': 0,
            'steps': [
                {'range': [0, max_val * 0.33], 'color': 'rgba(16,185,129,0.15)'},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': 'rgba(245,158,11,0.15)'},
                {'range': [max_val * 0.66, max_val], 'color': 'rgba(239,68,68,0.15)'}
            ],
            'threshold': {
                'line': {'color': '#818cf8', 'width': 3},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0'},
        height=220,
        margin=dict(l=30, r=30, t=40, b=10)
    )
    return fig


def render_footer():
    st.markdown("""
    <div class="app-footer">
        <p>🤖 AI Debt Collection Agents v2.0 — Powered by Multi-Agent AI Systems</p>
        <p>Built with ❤️ using Streamlit · FDCPA & TCPA Compliant</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# CACHING THE AI SYSTEM
# =========================================================
@st.cache_resource(show_spinner=False)
def load_ai_system():
    return DebtCollectionAPI()

api = load_ai_system()

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if 'nav_page' not in st.session_state:
    st.session_state.nav_page = "System Setup & Training"
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'export_format' not in st.session_state:
    st.session_state.export_format = 'csv'

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    # Logo / Brand
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 0.5rem 0;">
        <div style="font-size:2.5rem; margin-bottom:0.25rem;">🤖</div>
        <div style="font-size:1.1rem; font-weight:800;
             background:linear-gradient(135deg,#6366f1,#8b5cf6);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;
             background-clip:text; letter-spacing:-0.02em;">
            DebtCollect AI
        </div>
        <div style="font-size:0.7rem; color:#94a3b8; margin-top:0.15rem;">
            Multi-Agent Collection Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    st.markdown('<p style="font-size:0.7rem; font-weight:700; text-transform:uppercase; '
                'letter-spacing:0.1em; color:#64748b; margin-bottom:0.5rem;">📍 Navigation</p>',
                unsafe_allow_html=True)

    nav_items = {
        "System Setup & Training": "⚙️",
        "Dashboard Overview": "📊",
        "Debtor Analysis Profiler": "🔍",
        "Batch Priority Analysis": "📑",
        "Compliance Checker": "⚖️",
        "Analytics & Reports": "📈",
        "Settings": "🔧"
    }

    page = st.radio(
        "Navigate",
        list(nav_items.keys()),
        format_func=lambda x: f"{nav_items[x]}  {x}",
        key="nav_page",
        label_visibility="collapsed"
    )

    st.markdown("---")

    # System Status Panel
    st.markdown('<p style="font-size:0.7rem; font-weight:700; text-transform:uppercase; '
                'letter-spacing:0.1em; color:#64748b; margin-bottom:0.5rem;">🖥️ System Status</p>',
                unsafe_allow_html=True)

    status = api.get_status()

    if status['system_initialized']:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2);
                    border-radius:10px; padding:0.9rem;">
            <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.6rem;">
                <span class="badge badge-success">● ONLINE</span>
            </div>
            <div style="font-size:0.75rem; color:#94a3b8; line-height:1.8;">
                👥 Debtors: <strong style="color:#e2e8f0">{status['data_stats'].get('debtors_loaded', 0):,}</strong><br>
                💬 Communications: <strong style="color:#e2e8f0">{status['data_stats'].get('communications_loaded', 0):,}</strong><br>
                🕐 Last Updated: <strong style="color:#e2e8f0">{datetime.now().strftime('%H:%M')}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2);
                    border-radius:10px; padding:0.9rem;">
            <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.4rem;">
                <span class="badge badge-danger">● OFFLINE</span>
            </div>
            <div style="font-size:0.75rem; color:#94a3b8;">
                Train models to activate the system.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Actions
    st.markdown('<p style="font-size:0.7rem; font-weight:700; text-transform:uppercase; '
                'letter-spacing:0.1em; color:#64748b; margin-bottom:0.5rem;">⚡ Quick Actions</p>',
                unsafe_allow_html=True)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    if api.initialized:
        if st.button("📥 Export All Data", use_container_width=True):
            df = api.orchestrator.debtor_data
            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"debtors_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )


# =========================================================
# GATE: require initialization for most pages
# =========================================================
if not api.initialized and page not in ("System Setup & Training", "Settings"):
    render_page_header("⚠️", "System Not Initialized",
                       "The AI agents need to be trained before you can use this module. "
                       "Please go to System Setup & Training to get started.")

    st.markdown("""
    <div class="custom-alert alert-warning">
        <span>⚠️</span>
        <div>
            <strong>Action Required:</strong> Navigate to
            <em>System Setup &amp; Training</em> in the sidebar to initialize and train
            the multi-agent AI system.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Go to Setup", type="primary", use_container_width=True):
            st.session_state.nav_page = "System Setup & Training"
            st.rerun()
    st.stop()

# =========================================================
# PAGE: SYSTEM SETUP & TRAINING
# =========================================================
if page == "System Setup & Training":
    render_page_header(
        "⚙️", "System Setup & Training",
        "Initialize and train the multi-agent AI system. Choose synthetic data for a quick start "
        "or upload your own agency data for customized models."
    )

    # Training Progress Steps
    step1_class = "completed" if api.initialized else "active"
    step2_class = "completed" if api.initialized else "pending"
    step3_class = "completed" if api.initialized else "pending"
    step4_class = "completed" if api.initialized else "pending"
    conn_class = "completed" if api.initialized else ""

    st.markdown(f"""
    <div class="progress-container animate-in">
        <div class="progress-step">
            <div class="step-circle {step1_class}">1</div>
            <span class="step-label">Load Data</span>
        </div>
        <div class="step-connector {conn_class}"></div>
        <div class="progress-step">
            <div class="step-circle {step2_class}">2</div>
            <span class="step-label">Train Models</span>
        </div>
        <div class="step-connector {conn_class}"></div>
        <div class="progress-step">
            <div class="step-circle {step3_class}">3</div>
            <span class="step-label">Validate</span>
        </div>
        <div class="step-connector {conn_class}"></div>
        <div class="progress-step">
            <div class="step-circle {step4_class}">4</div>
            <span class="step-label">Deploy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🧬 Synthetic Data Generator", "📁 Upload Custom Data"])

    # --- TAB 1: SYNTHETIC ---
    with tab1:
        st.markdown("""
        <div class="section-card">
            <h3>🧬 Generate Synthetic Training Data</h3>
            <p style="color:var(--text-secondary); font-size:0.88rem;">
                Use the built-in data generator to simulate realistic debtor profiles and
                communication records. Ideal for demos, testing, and initial deployment.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            n_debtors = st.slider("📊 Number of Debtor Profiles", 100, 2000, 500, step=100,
                                  help="More profiles = better model accuracy but longer training.")
        with col2:
            n_comms = st.slider("💬 Number of Communications", 500, 5000, 1000, step=100,
                                help="Communication records for NLP and sentiment models.")

        # Resource estimate
        est_time = max(30, (n_debtors + n_comms) // 50)
        est_mem = max(50, (n_debtors + n_comms) // 20)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="analysis-panel" style="text-align:center;">
                <h4>⏱️ Est. Training Time</h4>
                <div style="font-size:1.3rem; font-weight:700; color:#818cf8;">{est_time}s</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="analysis-panel" style="text-align:center;">
                <h4>💾 Est. Memory Usage</h4>
                <div style="font-size:1.3rem; font-weight:700; color:#34d399;">{est_mem}MB</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="analysis-panel" style="text-align:center;">
                <h4>🤖 Models to Train</h4>
                <div style="font-size:1.3rem; font-weight:700; color:#fbbf24;">4 Agents</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-alert alert-info" style="margin-top:0.5rem;">
            <span>💡</span>
            <div>
                <strong>Cloud Deployment Tip:</strong> If using Streamlit Community Cloud
                (free tier), keep dataset sizes under 1,500 to avoid memory issues.
            </div>
        </div>
        """, unsafe_allow_html=True)

        training_successful = False
        if st.button("🚀 Initialize & Train System", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            steps = [
                ("Generating synthetic debtor profiles...", 15),
                ("Generating communication records...", 30),
                ("Training Risk Assessment Agent...", 50),
                ("Training NLP & Sentiment Agent...", 65),
                ("Training Payment Prediction Agent...", 80),
                ("Training RL Strategy Agent...", 90),
                ("Validating and deploying models...", 95),
            ]

            try:
                import threading, queue
                result_queue = queue.Queue()

                def train_in_background():
                    try:
                        api.setup(n_training_debtors=n_debtors, n_training_comms=n_comms)
                        result_queue.put(("success", None))
                    except Exception as e:
                        result_queue.put(("error", str(e)))

                thread = threading.Thread(target=train_in_background)
                thread.start()

                # Animate progress while training runs
                for label, pct in steps:
                    status_text.markdown(f"""
                    <div class="custom-alert alert-info">
                        <span class="pulse">⏳</span> <strong>{label}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    progress_bar.progress(pct)
                    time.sleep(max(1, est_time / len(steps) * 0.3))
                    if not thread.is_alive():
                        break

                thread.join()
                progress_bar.progress(100)

                res_status, res_err = result_queue.get_nowait()
                if res_status == "success":
                    training_successful = True
                    status_text.markdown("""
                    <div class="custom-alert alert-success">
                        <span>✅</span> <strong>All models trained and deployed successfully!</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    status_text.empty()
                    st.error(f"Training failed: {res_err}")

            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")

        if training_successful:
            time.sleep(1.5)
            st.session_state.nav_page = "Dashboard Overview"
            st.rerun()

    # --- TAB 2: CUSTOM DATA ---
    with tab2:
        st.markdown("""
        <div class="section-card">
            <h3>📁 Upload Your Agency Data</h3>
            <p style="color:var(--text-secondary); font-size:0.88rem;">
                Train the AI on your real debtor data for production-grade accuracy.
                Supports CSV and Excel formats.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Data Format Requirements", expanded=False):
            st.markdown("""
            **Debtors File** must contain these columns:
            | Column | Type | Description |
            |--------|------|-------------|
            | `debtor_id` | string | Unique debt identifier |
            | `first_name` | string | Debtor first name |
            | `last_name` | string | Debtor last name |
            | `total_debt` | float | Original debt amount |
            | `remaining_balance` | float | Current outstanding balance |
            | `days_past_due` | int | Days since last payment |
            | `credit_score` | int | Credit score (300-850) |
            | `income_estimate` | float | Estimated annual income |
            | `response_rate` | float | Historical response rate (0-1) |
            | `status` | string | Current account status |
            | `will_pay_30_days` | int | Target variable (0 or 1) |

            **Communications File** must contain:
            | Column | Type | Description |
            |--------|------|-------------|
            | `comm_id` | string | Unique communication ID |
            | `debtor_id` | string | Matching debtor ID |
            | `text` | string | Message transcript |
            | `intent` | string | e.g., reluctant, willing_to_pay |
            | `sentiment` | string | positive, negative, neutral |
            """)

        c1, c2 = st.columns(2)
        with c1:
            debtor_file = st.file_uploader(
                "📊 Upload Debtors Data",
                type=['csv', 'xlsx'],
                help="CSV or Excel with debtor profiles"
            )
            if debtor_file:
                st.markdown(f"""
                <div class="custom-alert alert-success">
                    <span>✅</span> <strong>{debtor_file.name}</strong> loaded
                    ({debtor_file.size / 1024:.1f} KB)
                </div>
                """, unsafe_allow_html=True)

        with c2:
            comm_file = st.file_uploader(
                "💬 Upload Communications Data",
                type=['csv', 'xlsx'],
                help="CSV or Excel with communication records"
            )
            if comm_file:
                st.markdown(f"""
                <div class="custom-alert alert-success">
                    <span>✅</span> <strong>{comm_file.name}</strong> loaded
                    ({comm_file.size / 1024:.1f} KB)
                </div>
                """, unsafe_allow_html=True)

        # Preview uploaded data
        if debtor_file or comm_file:
            with st.expander("👁️ Preview Uploaded Data", expanded=False):
                if debtor_file:
                    st.markdown("**Debtors Preview:**")
                    load_func = pd.read_csv if debtor_file.name.endswith('.csv') else pd.read_excel
                    preview_df = load_func(debtor_file)
                    st.dataframe(preview_df.head(5), use_container_width=True)
                    debtor_file.seek(0)  # reset pointer
                    st.caption(f"Shape: {preview_df.shape[0]} rows × {preview_df.shape[1]} columns")

                if comm_file:
                    st.markdown("**Communications Preview:**")
                    load_func = pd.read_csv if comm_file.name.endswith('.csv') else pd.read_excel
                    preview_df = load_func(comm_file)
                    st.dataframe(preview_df.head(5), use_container_width=True)
                    comm_file.seek(0)
                    st.caption(f"Shape: {preview_df.shape[0]} rows × {preview_df.shape[1]} columns")

        training_successful_upload = False
        if st.button("🚀 Train with Uploaded Data", type="primary", use_container_width=True):
            if debtor_file and comm_file:
                with st.spinner("Processing files and training AI agents..."):
                    try:
                        load_func_d = pd.read_csv if debtor_file.name.endswith('.csv') else pd.read_excel
                        load_func_c = pd.read_csv if comm_file.name.endswith('.csv') else pd.read_excel
                        df_debtors = load_func_d(debtor_file)
                        df_comms = load_func_c(comm_file)

                        orig_gen_d = api.orchestrator.data_generator.generate_debtor_profiles
                        orig_gen_c = api.orchestrator.data_generator.generate_communication_data
                        api.orchestrator.data_generator.generate_debtor_profiles = lambda n: df_debtors
                        api.orchestrator.data_generator.generate_communication_data = lambda n: df_comms

                        api.setup(n_training_debtors=len(df_debtors), n_training_comms=len(df_comms))

                        api.orchestrator.data_generator.generate_debtor_profiles = orig_gen_d
                        api.orchestrator.data_generator.generate_communication_data = orig_gen_c
                        training_successful_upload = True
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.markdown("""
                        <div class="custom-alert alert-warning">
                            <span>⚠️</span> Ensure your files match the required column names above.
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="custom-alert alert-warning">
                    <span>⚠️</span> Please upload <strong>both</strong> files to proceed.
                </div>
                """, unsafe_allow_html=True)

        if training_successful_upload:
            st.markdown("""
            <div class="custom-alert alert-success">
                <span>🎉</span> <strong>Models trained on your data!</strong> Redirecting to Dashboard...
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.5)
            st.session_state.nav_page = "Dashboard Overview"
            st.rerun()

    render_footer()

# =========================================================
# PAGE: DASHBOARD OVERVIEW
# =========================================================
elif page == "Dashboard Overview":
    render_page_header(
        "📊", "Agency Dashboard",
        "Real-time overview of your debt collection portfolio performance, debtor demographics, "
        "and key collection metrics."
    )

    df = api.orchestrator.debtor_data

    # ── KPI Row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("👥", "Total Debtors", f"{len(df):,}",
                           delta="Active accounts", delta_positive=True, color_class="purple")
    with c2:
        render_metric_card("💰", "Total Outstanding", f"${df['remaining_balance'].sum():,.0f}",
                           delta=f"Avg ${df['remaining_balance'].mean():,.0f}/debtor",
                           delta_positive=False, color_class="blue")
    with c3:
        render_metric_card("📊", "Avg Credit Score", f"{int(df['credit_score'].mean())}",
                           delta=f"Range: {int(df['credit_score'].min())}-{int(df['credit_score'].max())}",
                           delta_positive=True, color_class="green")
    with c4:
        render_metric_card("📞", "Avg Response Rate", f"{df['response_rate'].mean():.1%}",
                           delta="Across all channels", delta_positive=True, color_class="amber")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Secondary KPI Row ──
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        high_risk = len(df[df['days_past_due'] > 90]) if 'days_past_due' in df.columns else 0
        render_metric_card("⚠️", "High Risk (>90 DPD)", f"{high_risk:,}",
                           delta=f"{high_risk / len(df) * 100:.1f}% of portfolio",
                           delta_positive=False, color_class="purple")
    with c6:
        avg_dpd = int(df['days_past_due'].mean()) if 'days_past_due' in df.columns else 0
        render_metric_card("📅", "Avg Days Past Due", f"{avg_dpd}",
                           color_class="blue")
    with c7:
        total_debt = df['total_debt'].sum() if 'total_debt' in df.columns else 0
        recovered = total_debt - df['remaining_balance'].sum()
        recovery_rate = recovered / total_debt * 100 if total_debt > 0 else 0
        render_metric_card("📈", "Recovery Rate", f"{recovery_rate:.1f}%",
                           delta=f"${recovered:,.0f} recovered", delta_positive=True,
                           color_class="green")
    with c8:
        unique_types = df['debt_type'].nunique() if 'debt_type' in df.columns else 0
        render_metric_card("🏷️", "Debt Categories", f"{unique_types}",
                           color_class="amber")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Charts Row 1 ──
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("""
        <div class="section-card"><h3>📊 Debt Status Distribution</h3></div>
        """, unsafe_allow_html=True)
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        status_counts['Status'] = status_counts['Status'].str.replace('_', ' ').str.title()

        fig1 = px.pie(
            status_counts, names='Status', values='Count',
            hole=0.45,
            color_discrete_sequence=['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd',
                                     '#34d399', '#fbbf24', '#f87171']
        )
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            legend=dict(font=dict(size=11)),
            margin=dict(l=20, r=20, t=20, b=20),
            height=350
        )
        fig1.update_traces(textinfo='percent+label', textfont_size=11)
        st.plotly_chart(fig1, use_container_width=True)

    with c_right:
        st.markdown("""
        <div class="section-card"><h3>🏷️ Debt Types Breakdown</h3></div>
        """, unsafe_allow_html=True)
        if 'debt_type' in df.columns:
            type_counts = df['debt_type'].value_counts().reset_index()
            type_counts.columns = ['Debt Type', 'Count']
            type_counts['Debt Type'] = type_counts['Debt Type'].str.replace('_', ' ').str.title()

            fig2 = px.bar(
                type_counts, x='Debt Type', y='Count',
                color='Count',
                color_continuous_scale=['#6366f1', '#8b5cf6', '#a78bfa']
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(gridcolor='#2d2f3e'),
                yaxis=dict(gridcolor='#2d2f3e'),
                margin=dict(l=20, r=20, t=20, b=20),
                height=350,
                showlegend=False,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Charts Row 2 ──
    c_left2, c_right2 = st.columns(2)

    with c_left2:
        st.markdown("""
        <div class="section-card"><h3>💳 Credit Score Distribution</h3></div>
        """, unsafe_allow_html=True)
        fig3 = px.histogram(
            df, x='credit_score', nbins=30,
            color_discrete_sequence=['#6366f1']
        )
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(title='Credit Score', gridcolor='#2d2f3e'),
            yaxis=dict(title='Count', gridcolor='#2d2f3e'),
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            bargap=0.05
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c_right2:
        st.markdown("""
        <div class="section-card"><h3>📅 Days Past Due vs Balance</h3></div>
        """, unsafe_allow_html=True)
        fig4 = px.scatter(
            df.sample(min(200, len(df))),
            x='days_past_due', y='remaining_balance',
            color='credit_score',
            size='remaining_balance',
            color_continuous_scale='Viridis',
            opacity=0.7
        )
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(title='Days Past Due', gridcolor='#2d2f3e'),
            yaxis=dict(title='Remaining Balance ($)', gridcolor='#2d2f3e'),
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Top Debtors Table ──
    st.markdown("""
    <div class="section-card"><h3>🔝 Top 10 Highest Outstanding Balances</h3></div>
    """, unsafe_allow_html=True)

    top_debtors = df.nlargest(10, 'remaining_balance')[
        ['debtor_id', 'first_name', 'last_name', 'remaining_balance',
         'days_past_due', 'credit_score', 'status']
    ].copy()
    top_debtors['remaining_balance'] = top_debtors['remaining_balance'].apply(lambda x: f"${x:,.2f}")
    top_debtors['status'] = top_debtors['status'].str.replace('_', ' ').str.title()
    top_debtors.columns = ['ID', 'First Name', 'Last Name', 'Balance', 'DPD', 'Credit Score', 'Status']

    st.dataframe(top_debtors, use_container_width=True, hide_index=True)

    render_footer()

# =========================================================
# PAGE: DEBTOR ANALYSIS PROFILER
# =========================================================
elif page == "Debtor Analysis Profiler":
    render_page_header(
        "🔍", "Comprehensive Debtor Profiler",
        "Deep-dive analysis of individual debtors using all AI agents. Get risk assessment, "
        "payment predictions, optimal strategies, and compliance checks in one view."
    )

    df = api.orchestrator.debtor_data
    debtor_ids = df['debtor_id'].tolist()

    col_profile, col_analysis = st.columns([1, 2.5])

    with col_profile:
        selected_id = st.selectbox("🔎 Search Debtor", debtor_ids,
                                   help="Select or search for a debtor ID")
        debtor_row = df[df['debtor_id'] == selected_id].iloc[0]

        first_name = debtor_row.get('first_name', 'Unknown')
        last_name = debtor_row.get('last_name', '')
        initials = f"{first_name[0]}{last_name[0]}" if last_name else first_name[0]

        st.markdown(f"""
        <div class="profile-card animate-in">
            <div class="profile-avatar">{initials.upper()}</div>
            <div class="profile-name">{first_name} {last_name}</div>
            <div class="profile-id">ID: {selected_id}</div>
            <div style="margin-bottom:0.75rem;">
                {render_badge(str(debtor_row.get('status', '')).replace('_', ' ').title(), 'purple')}
            </div>
            <div class="profile-stats">
                <div class="profile-stat">
                    <div class="profile-stat-label">Balance</div>
                    <div class="profile-stat-value">${debtor_row.get('remaining_balance', 0):,.0f}</div>
                </div>
                <div class="profile-stat">
                    <div class="profile-stat-label">Days Past Due</div>
                    <div class="profile-stat-value">{debtor_row.get('days_past_due', 0)}</div>
                </div>
                <div class="profile-stat">
                    <div class="profile-stat-label">Credit Score</div>
                    <div class="profile-stat-value">{debtor_row.get('credit_score', 'N/A')}</div>
                </div>
                <div class="profile-stat">
                    <div class="profile-stat-label">Response Rate</div>
                    <div class="profile-stat-value">{debtor_row.get('response_rate', 0):.0%}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Additional debtor info
        with st.expander("📋 Full Profile Details"):
            for col_name in df.columns:
                if col_name not in ('first_name', 'last_name', 'debtor_id'):
                    val = debtor_row.get(col_name, 'N/A')
                    label = col_name.replace('_', ' ').title()
                    if isinstance(val, float):
                        val = f"{val:,.2f}"
                    st.markdown(f"**{label}:** {val}")

    with col_analysis:
        st.markdown("""
        <div class="section-card">
            <h3>💬 Simulate Incoming Communication</h3>
        </div>
        """, unsafe_allow_html=True)

        custom_msg = st.text_area(
            "Debtor's recent email/call transcript (optional)",
            placeholder="e.g., I lost my job and cannot afford payments right now. "
                        "Please give me some time to get back on my feet.",
            height=100,
            label_visibility="collapsed"
        )

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            run_analysis = st.button("🚀 Run Multi-Agent Analysis", type="primary",
                                     use_container_width=True)
        with col_btn2:
            quick_analysis = st.button("⚡ Quick Scan", use_container_width=True)

        if run_analysis or quick_analysis:
            with st.spinner("AI agents are analyzing..."):
                analysis = api.analyze_debtor(
                    debtor_id=selected_id,
                    message=custom_msg if custom_msg else None
                )

            if 'error' in analysis:
                st.error(analysis['error'])
            else:
                # Store in history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'debtor_id': selected_id,
                    'priority_score': analysis['priority_score'],
                    'risk_level': analysis['agents_results']['risk_assessment']['risk_level']
                })

                # Priority Score Gauge
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                g1, g2, g3 = st.columns(3)
                with g1:
                    fig_gauge = create_gauge_chart(
                        analysis['priority_score'],
                        "Priority Score"
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                with g2:
                    pay_prob = analysis['agents_results']['payment_prediction']['will_pay_probability'] * 100
                    fig_gauge2 = create_gauge_chart(pay_prob, "Payment Probability")
                    st.plotly_chart(fig_gauge2, use_container_width=True)
                with g3:
                    risk_level = analysis['agents_results']['risk_assessment']['risk_level']
                    risk_val = {'low': 25, 'medium': 55, 'high': 85}.get(risk_level.lower(), 50)
                    fig_gauge3 = create_gauge_chart(risk_val, "Risk Level")
                    st.plotly_chart(fig_gauge3, use_container_width=True)

                # Detailed tabs
                t1, t2, t3, t4 = st.tabs([
                    "⚠️ Risk & Payment", "🎯 Strategy",
                    "💬 Communication", "⚖️ Compliance"
                ])

                with t1:
                    risk = analysis['agents_results']['risk_assessment']
                    pay = analysis['agents_results']['payment_prediction']

                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown("""
                        <div class="analysis-panel"><h4>Risk Assessment</h4></div>
                        """, unsafe_allow_html=True)
                        st.markdown(render_risk_indicator(risk['risk_level']),
                                    unsafe_allow_html=True)
                        st.markdown(f"**Risk Score:** {risk.get('risk_score', 'N/A')}")
                        st.markdown(f"**Payment Probability:** {risk.get('payment_probability', 0):.1%}")

                        if 'risk_factors' in risk:
                            st.markdown("**Key Risk Factors:**")
                            for factor in risk.get('risk_factors', []):
                                st.markdown(f"- {factor}")

                    with rc2:
                        st.markdown("""
                        <div class="analysis-panel"><h4>Payment Prediction</h4></div>
                        """, unsafe_allow_html=True)
                        st.metric("Will Pay (30 days)", f"{pay['will_pay_probability']:.1%}")
                        st.metric("Expected Recovery",
                                  f"${pay.get('expected_value', 0):,.2f}")
                        if 'confidence_interval' in pay:
                            ci = pay['confidence_interval']
                            st.caption(f"95% CI: ${ci.get('low', 0):,.0f} – "
                                       f"${ci.get('high', 0):,.0f}")

                with t2:
                    strat = analysis['agents_results']['strategy']
                    sc1, sc2 = st.columns(2)

                    with sc1:
                        st.markdown("""
                        <div class="analysis-panel"><h4>Recommended Action</h4></div>
                        """, unsafe_allow_html=True)
                        recommended = strat['recommended_strategy'].replace('_', ' ').title()
                        st.markdown(f"""
                        <div class="risk-indicator risk-medium" style="margin-bottom:1rem;">
                            🎯 {recommended}
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"**Primary Channel:** "
                                    f"{strat['recommended_channel']['primary'].title()}")
                        if 'secondary' in strat['recommended_channel']:
                            st.markdown(f"**Secondary Channel:** "
                                        f"{strat['recommended_channel']['secondary'].title()}")
                        st.markdown(f"**Best Day:** "
                                    f"{strat['recommended_timing']['best_day']}")
                        st.markdown(f"**Best Time:** "
                                    f"{strat['recommended_timing']['best_time']}")

                    with sc2:
                        st.markdown("""
                        <div class="analysis-panel"><h4>Strategy Rankings</h4></div>
                        """, unsafe_allow_html=True)
                        ranking_df = pd.DataFrame(strat['strategy_ranking'])
                        if not ranking_df.empty:
                            fig_strat = px.bar(
                                ranking_df.head(5),
                                x='q_value' if 'q_value' in ranking_df.columns else ranking_df.columns[1],
                                y='strategy' if 'strategy' in ranking_df.columns else ranking_df.columns[0],
                                orientation='h',
                                color_discrete_sequence=['#6366f1']
                            )
                            fig_strat.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#e2e8f0', size=11),
                                xaxis=dict(gridcolor='#2d2f3e'),
                                yaxis=dict(gridcolor='#2d2f3e'),
                                margin=dict(l=10, r=10, t=10, b=10),
                                height=250,
                                showlegend=False
                            )
                            st.plotly_chart(fig_strat, use_container_width=True)
                        else:
                            st.dataframe(ranking_df, use_container_width=True)

                with t3:
                    if custom_msg:
                        comm = analysis['agents_results'].get('communication', {})
                        cc1, cc2 = st.columns(2)

                        with cc1:
                            st.markdown("""
                            <div class="analysis-panel"><h4>Sentiment Analysis</h4></div>
                            """, unsafe_allow_html=True)
                            sentiment = comm.get('vader_category', 'N/A').title()
                            sentiment_badge = 'success' if sentiment.lower() == 'positive' \
                                else ('danger' if sentiment.lower() == 'negative' else 'warning')
                            st.markdown(render_badge(f"Sentiment: {sentiment}", sentiment_badge),
                                        unsafe_allow_html=True)

                            intent = comm.get('intent', {})
                            intent_class = intent.get('classification', 'N/A').replace('_', ' ').title()
                            st.markdown(f"**Intent:** {intent_class}")
                            st.markdown(f"**Confidence:** "
                                        f"{intent.get('confidence', 0):.1%}"
                                        if 'confidence' in intent else "")

                            flags = comm.get('compliance_flags', [])
                            if flags:
                                st.markdown("**⚠️ Regulatory Flags:**")
                                for flag in flags:
                                    st.markdown(f"""
                                    <div class="violation-card">
                                        <div class="violation-rule">🚩 {flag}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                        with cc2:
                            st.markdown("""
                            <div class="analysis-panel"><h4>AI Generated Response</h4></div>
                            """, unsafe_allow_html=True)
                            gen_resp = analysis['agents_results'].get(
                                'generated_response', {}).get('response', '')
                            st.text_area("Draft Response", gen_resp, height=200,
                                         label_visibility="collapsed")

                            r1, r2 = st.columns(2)
                            with r1:
                                st.button("📋 Copy", use_container_width=True)
                            with r2:
                                st.button("✏️ Edit & Send", use_container_width=True)
                    else:
                        st.markdown("""
                        <div class="custom-alert alert-info">
                            <span>💡</span> Enter a communication transcript above to
                            activate NLP analysis and auto-response generation.
                        </div>
                        """, unsafe_allow_html=True)

                with t4:
                    comp = analysis['agents_results']['compliance']
                    if comp['is_approved']:
                        st.markdown(f"""
                        <div class="custom-alert alert-success">
                            <span>✅</span>
                            <div><strong>Compliant</strong> — {comp['recommendation']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="custom-alert alert-danger">
                            <span>❌</span>
                            <div><strong>Non-Compliant</strong> — {comp['recommendation']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        for v in comp.get('violations', []):
                            severity_badge = 'danger' if v.get('severity') == 'critical' else 'warning'
                            st.markdown(f"""
                            <div class="violation-card">
                                <div class="violation-rule">
                                    {render_badge(v.get('severity', 'warning').upper(), severity_badge)}
                                    {v['rule']}
                                </div>
                                <div class="violation-desc">{v['description']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Export analysis
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                with st.expander("📥 Export Analysis Results"):
                    export_data = json.dumps(analysis, indent=2, default=str)
                    st.download_button(
                        "⬇️ Download JSON Report",
                        data=export_data,
                        file_name=f"analysis_{selected_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

    render_footer()

# =========================================================
# PAGE: BATCH PRIORITY ANALYSIS
# =========================================================
elif page == "Batch Priority Analysis":
    render_page_header(
        "📑", "Batch Queue Optimization",
        "Analyze multiple debtors simultaneously to create a priority-ranked collection queue. "
        "Optimize agent workload allocation based on AI-driven scoring."
    )

    # Controls
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        batch_size = st.slider("📊 Debtors to Analyze", 10, 100, 25, step=5)
    with ctrl2:
        sort_by = st.selectbox("Sort By", ["Priority Score", "Risk Level", "Payment Prob."])
    with ctrl3:
        filter_risk = st.multiselect("Filter Risk", ["HIGH", "MEDIUM", "LOW"],
                                     default=["HIGH", "MEDIUM", "LOW"])

    c_btn1, c_btn2 = st.columns([3, 1])
    with c_btn1:
        run_batch = st.button("🚀 Run Batch Optimization", type="primary",
                              use_container_width=True)
    with c_btn2:
        if st.session_state.batch_results:
            st.button("🗑️ Clear Results", use_container_width=True,
                       on_click=lambda: st.session_state.update(batch_results=None))

    if run_batch:
        progress = st.progress(0)
        status = st.empty()

        with st.spinner(f"Analyzing {batch_size} debtors across all AI agents..."):
            results = api.batch_analyze(n=batch_size)

            clean_results = []
            for i, r in enumerate(results):
                progress.progress((i + 1) / len(results))
                if 'error' not in r:
                    clean_results.append({
                        "Debtor ID": r['debtor_id'],
                        "Priority Score": r['priority_score'],
                        "Risk Level": r['agents_results']['risk_assessment']['risk_level'].upper(),
                        "Strategy": r['agents_results']['strategy']['recommended_strategy']
                            .replace('_', ' ').title(),
                        "Payment Prob.": f"{r['agents_results']['risk_assessment']['payment_probability']:.1%}",
                        "Payment Prob Num": r['agents_results']['risk_assessment']['payment_probability'],
                        "Compliance": "✅ Pass" if r['agents_results']['compliance']['is_approved'] else "❌ Fail",
                        "Channel": r['agents_results']['strategy']['recommended_channel']['primary'].title()
                    })

            st.session_state.batch_results = clean_results
            progress.progress(100)

    if st.session_state.batch_results:
        results_df = pd.DataFrame(st.session_state.batch_results)

        # Filter
        results_df = results_df[results_df['Risk Level'].isin(filter_risk)]

        # Sort
        if sort_by == "Priority Score":
            results_df = results_df.sort_values('Priority Score', ascending=False)
        elif sort_by == "Risk Level":
            risk_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            results_df = results_df.sort_values('Risk Level',
                                                key=lambda x: x.map(risk_order))
        elif sort_by == "Payment Prob.":
            results_df = results_df.sort_values('Payment Prob Num', ascending=True)

        # Summary metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            render_metric_card("📊", "Analyzed", f"{len(results_df)}",
                               color_class="purple")
        with mc2:
            high_count = len(results_df[results_df['Risk Level'] == 'HIGH'])
            render_metric_card("🔴", "High Risk", f"{high_count}",
                               delta=f"{high_count / len(results_df) * 100:.0f}%",
                               delta_positive=False, color_class="blue")
        with mc3:
            pass_count = len(results_df[results_df['Compliance'] == '✅ Pass'])
            render_metric_card("✅", "Compliant", f"{pass_count}",
                               delta=f"{pass_count / len(results_df) * 100:.0f}%",
                               delta_positive=True, color_class="green")
        with mc4:
            avg_priority = results_df['Priority Score'].mean()
            render_metric_card("🎯", "Avg Priority", f"{avg_priority:.0f}",
                               color_class="amber")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Visualization
        viz1, viz2 = st.columns(2)
        with viz1:
            st.markdown("""
            <div class="section-card"><h3>📊 Priority Distribution</h3></div>
            """, unsafe_allow_html=True)
            fig_hist = px.histogram(
                results_df, x='Priority Score', nbins=20,
                color='Risk Level',
                color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'}
            )
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(gridcolor='#2d2f3e'),
                yaxis=dict(gridcolor='#2d2f3e'),
                margin=dict(l=20, r=20, t=20, b=20),
                height=300, bargap=0.05
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with viz2:
            st.markdown("""
            <div class="section-card"><h3>🎯 Strategy Distribution</h3></div>
            """, unsafe_allow_html=True)
            strat_counts = results_df['Strategy'].value_counts().reset_index()
            strat_counts.columns = ['Strategy', 'Count']
            fig_strat = px.pie(
                strat_counts, names='Strategy', values='Count', hole=0.4,
                color_discrete_sequence=['#6366f1', '#8b5cf6', '#a78bfa',
                                         '#34d399', '#fbbf24', '#f87171']
            )
            fig_strat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                margin=dict(l=20, r=20, t=20, b=20),
                height=300
            )
            st.plotly_chart(fig_strat, use_container_width=True)

        # Results table
        st.markdown("""
        <div class="section-card"><h3>📋 Prioritized Collection Queue</h3></div>
        """, unsafe_allow_html=True)

        display_df = results_df.drop(columns=['Payment Prob Num'], errors='ignore')
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

        # Export
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                "📥 Export as CSV",
                data=csv_data,
                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c_exp2:
            json_data = display_df.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Export as JSON",
                data=json_data,
                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )

    render_footer()

# =========================================================
# PAGE: COMPLIANCE CHECKER
# =========================================================
elif page == "Compliance Checker":
    render_page_header(
        "⚖️", "FDCPA / TCPA Compliance Sandbox",
        "Test proposed collection actions against federal regulations before execution. "
        "Ensure every debtor interaction is fully compliant with FDCPA, TCPA, and SCRA rules."
    )

    col_flags, col_action = st.columns([1, 1.5])

    with col_flags:
        st.markdown("""
        <div class="section-card"><h3>🛡️ Debtor Protection Flags</h3></div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-alert alert-info" style="margin-bottom:1rem;">
            <span>ℹ️</span> Select all applicable flags for the debtor.
        </div>
        """, unsafe_allow_html=True)

        dnc = st.toggle("📵 Do Not Call Registry", value=False)
        cnd = st.toggle("🛑 Cease and Desist Requested", value=False)
        att = st.toggle("⚖️ Represented by Attorney", value=False)
        bnk = st.toggle("📋 Bankruptcy Filed", value=False)
        mil = st.toggle("🎖️ Active Military (SCRA)", value=False)

        st.markdown("---")

        st.markdown("**Contact Frequency (Today/This Week):**")
        cf1, cf2 = st.columns(2)
        with cf1:
            contacts_today = st.number_input("Today", 0, 10, 0)
        with cf2:
            contacts_week = st.number_input("This Week", 0, 30, 0)

    with col_action:
        st.markdown("""
        <div class="section-card"><h3>📤 Proposed Collection Action</h3></div>
        """, unsafe_allow_html=True)

        ac1, ac2 = st.columns(2)
        with ac1:
            channel = st.selectbox("📱 Communication Channel",
                                   ["phone", "email", "sms", "letter"],
                                   format_func=lambda x: f"{'📞' if x == 'phone' else '📧' if x == 'email' else '📱' if x == 'sms' else '✉️'} {x.title()}")
        with ac2:
            contact_time = st.time_input("🕐 Time of Contact",
                                         value=datetime.now().replace(hour=10, minute=0))

        msg = st.text_area(
            "📝 Message Content",
            placeholder="Enter the proposed message to send to the debtor...",
            height=150
        )

        # Quick Templates
        with st.expander("📋 Message Templates"):
            templates = {
                "Standard Reminder": "This is a reminder that your account has a balance of $X. "
                                     "Please contact us to discuss payment options.",
                "Payment Plan Offer": "We'd like to offer you a flexible payment plan to help "
                                      "resolve your outstanding balance. Please call us at your convenience.",
                "Final Notice": "This is a final notice regarding your outstanding debt. "
                                "Please contact our office within 30 days to resolve this matter.",
                "Aggressive (Test)": "Pay your debt now or we will seize your property and garnish your wages!"
            }
            selected_template = st.selectbox("Choose Template", list(templates.keys()))
            if st.button("📋 Use This Template"):
                msg = templates[selected_template]
                st.rerun()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if st.button("🔍 Run Compliance Check", type="primary", use_container_width=True):
        if not msg:
            st.markdown("""
            <div class="custom-alert alert-warning">
                <span>⚠️</span> Please enter a message to check compliance.
            </div>
            """, unsafe_allow_html=True)
        else:
            debtor_context = {
                'do_not_call': dnc, 'cease_and_desist': cnd,
                'represented_by_attorney': att, 'bankruptcy_filed': bnk,
                'active_military': mil, 'contacts_today': contacts_today,
                'contacts_this_week': contacts_week
            }
            action_context = {
                'channel': channel,
                'time': contact_time.strftime("%H:%M"),
                'message': msg
            }

            with st.spinner("Checking compliance..."):
                result = api.check_compliance(action_context, debtor_context)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            if result['compliance_status'] == 'compliant':
                st.markdown(f"""
                <div class="custom-alert alert-success" style="padding:1.5rem;">
                    <span style="font-size:2rem;">✅</span>
                    <div>
                        <strong style="font-size:1.1rem;">Fully Compliant</strong><br>
                        <span style="font-size:0.9rem;">This action meets all FDCPA, TCPA,
                        and SCRA requirements. Safe to proceed.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="section-card">
                    <h3>✅ Compliance Summary</h3>
                </div>
                """, unsafe_allow_html=True)

                checks = [
                    ("FDCPA Time Restrictions", "Compliant", "success"),
                    ("Do Not Call Compliance", "Compliant" if not dnc else "N/A", "success"),
                    ("Cease & Desist Compliance", "Compliant" if not cnd else "N/A", "success"),
                    ("Bankruptcy Stay", "Compliant" if not bnk else "N/A", "success"),
                    ("SCRA Protections", "Compliant" if not mil else "N/A", "success"),
                    ("Language & Tone", "Compliant", "success"),
                ]
                for check_name, check_status, badge_type in checks:
                    st.markdown(f"- {render_badge(check_status, badge_type)} **{check_name}**",
                                unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="custom-alert alert-danger" style="padding:1.5rem;">
                    <span style="font-size:2rem;">❌</span>
                    <div>
                        <strong style="font-size:1.1rem;">Non-Compliant — 
                        {result['compliance_status'].upper()}</strong><br>
                        <span style="font-size:0.9rem;">{result.get('recommendation', '')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if 'violations' in result:
                    st.markdown("""
                    <div class="section-card"><h3>🚨 Violations Detected</h3></div>
                    """, unsafe_allow_html=True)
                    for v in result['violations']:
                        st.markdown(f"""
                        <div class="violation-card">
                            <div class="violation-rule">
                                {render_badge(v.get('severity', 'warning').upper(),
                                              'danger' if v.get('severity') == 'critical' else 'warning')}
                                &nbsp; {v['rule']}
                            </div>
                            <div class="violation-desc">{v['description']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                if 'warnings' in result:
                    st.markdown("""
                    <div class="section-card"><h3>⚠️ Warnings</h3></div>
                    """, unsafe_allow_html=True)
                    for w in result['warnings']:
                        st.markdown(f"""
                        <div class="warning-card">
                            <div style="color:#fcd34d; font-weight:600;">⚠️ {w}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Export compliance result
            with st.expander("📥 Export Compliance Report"):
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'debtor_flags': debtor_context,
                    'proposed_action': action_context,
                    'result': result
                }
                st.download_button(
                    "⬇️ Download Report (JSON)",
                    data=json.dumps(report, indent=2, default=str),
                    file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )

    render_footer()

# =========================================================
# PAGE: ANALYTICS & REPORTS (NEW)
# =========================================================
elif page == "Analytics & Reports":
    render_page_header(
        "📈", "Analytics & Reports",
        "Comprehensive portfolio analytics, trend analysis, and performance insights "
        "to drive data-informed collection strategies."
    )

    df = api.orchestrator.debtor_data

    # Portfolio Overview
    st.markdown("""
    <div class="section-card"><h3>📊 Portfolio Analytics</h3></div>
    """, unsafe_allow_html=True)

    tab_a1, tab_a2, tab_a3 = st.tabs([
        "📊 Distribution Analysis", "📈 Correlation Analysis", "📋 Segment Analysis"
    ])

    with tab_a1:
        a1, a2 = st.columns(2)

        with a1:
            st.markdown("**Balance Distribution by Status**")
            fig_box = px.box(
                df, x='status', y='remaining_balance',
                color='status',
                color_discrete_sequence=['#6366f1', '#8b5cf6', '#a78bfa',
                                         '#34d399', '#fbbf24']
            )
            fig_box.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(gridcolor='#2d2f3e', title='Status'),
                yaxis=dict(gridcolor='#2d2f3e', title='Remaining Balance ($)'),
                margin=dict(l=20, r=20, t=20, b=20),
                height=350, showlegend=False
            )
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)

        with a2:
            st.markdown("**Income vs Remaining Balance**")
            if 'income_estimate' in df.columns:
                fig_scatter = px.scatter(
                    df.sample(min(300, len(df))),
                    x='income_estimate', y='remaining_balance',
                    color='status', size='days_past_due',
                    opacity=0.6,
                    color_discrete_sequence=['#6366f1', '#34d399', '#f59e0b',
                                             '#ef4444', '#8b5cf6']
                )
                fig_scatter.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    xaxis=dict(gridcolor='#2d2f3e', title='Income Estimate ($)'),
                    yaxis=dict(gridcolor='#2d2f3e', title='Remaining Balance ($)'),
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=350
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_a2:
        st.markdown("**Feature Correlation Heatmap**")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        display_cols = [c for c in numeric_cols
                        if c in ('remaining_balance', 'total_debt', 'days_past_due',
                                 'credit_score', 'income_estimate', 'response_rate',
                                 'will_pay_30_days')]
        if display_cols:
            corr = df[display_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig_corr.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0', size=11),
                margin=dict(l=20, r=20, t=20, b=20),
                height=450
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab_a3:
        st.markdown("**Debtor Segmentation by Risk Tier**")

        # Create risk tiers
        df_seg = df.copy()
        df_seg['risk_tier'] = pd.cut(
            df_seg['days_past_due'],
            bins=[0, 30, 60, 90, 180, float('inf')],
            labels=['Current (0-30)', 'Early (31-60)',
                    'Mid (61-90)', 'Late (91-180)', 'Severe (180+)']
        )

        seg_summary = df_seg.groupby('risk_tier', observed=True).agg({
            'debtor_id': 'count',
            'remaining_balance': ['sum', 'mean'],
            'credit_score': 'mean',
            'response_rate': 'mean'
        }).round(2)

        seg_summary.columns = ['Count', 'Total Balance', 'Avg Balance',
                               'Avg Credit Score', 'Avg Response Rate']
        seg_summary = seg_summary.reset_index()

        # Display
        st.dataframe(seg_summary, use_container_width=True, hide_index=True)

        # Segment visualization
        fig_seg = px.bar(
            seg_summary, x='risk_tier', y='Total Balance',
            color='Count',
            color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
            text='Count'
        )
        fig_seg.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(title='Risk Tier', gridcolor='#2d2f3e'),
            yaxis=dict(title='Total Balance ($)', gridcolor='#2d2f3e'),
            margin=dict(l=20, r=20, t=20, b=20),
            height=350,
            coloraxis_showscale=False
        )
        fig_seg.update_traces(textposition='outside')
        st.plotly_chart(fig_seg, use_container_width=True)

    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("""
        <div class="section-card"><h3>📜 Recent Analysis History</h3></div>
        """, unsafe_allow_html=True)
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

    render_footer()

# =========================================================
# PAGE: SETTINGS (NEW)
# =========================================================
elif page == "Settings":
    render_page_header(
        "🔧", "Settings & Configuration",
        "Customize application behavior, manage data, and configure system preferences."
    )

    tab_s1, tab_s2, tab_s3 = st.tabs([
        "⚙️ General Settings", "🗃️ Data Management", "ℹ️ About"
    ])

    with tab_s1:
        st.markdown("""
        <div class="section-card"><h3>⚙️ Display Preferences</h3></div>
        """, unsafe_allow_html=True)

        st.session_state.export_format = st.selectbox(
            "Default Export Format", ['csv', 'json', 'xlsx'],
            index=['csv', 'json', 'xlsx'].index(st.session_state.export_format)
        )

        st.markdown("""
        <div class="section-card"><h3>🔔 Notification Preferences</h3></div>
        """, unsafe_allow_html=True)

        st.toggle("Enable high-risk alerts", value=True)
        st.toggle("Enable compliance warnings", value=True)
        st.toggle("Enable training completion notifications", value=True)

    with tab_s2:
        st.markdown("""
        <div class="section-card"><h3>🗃️ Data Management</h3></div>
        """, unsafe_allow_html=True)

        if api.initialized:
            df = api.orchestrator.debtor_data
            st.markdown(f"""
            <div class="custom-alert alert-info">
                <span>ℹ️</span>
                <div>
                    <strong>Current Dataset:</strong>
                    {len(df):,} debtors loaded<br>
                    <strong>Memory Usage:</strong> ~{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🗑️ Clear All Data & Reset System", type="primary"):
                st.cache_resource.clear()
                st.session_state.analysis_history = []
                st.session_state.batch_results = None
                st.rerun()

            st.markdown("---")
            st.markdown("**Export Full Dataset:**")
            c1, c2 = st.columns(2)
            with c1:
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv,
                                   f"full_data_{datetime.now().strftime('%Y%m%d')}.csv",
                                   "text/csv", use_container_width=True)
            with c2:
                json_data = df.to_json(orient='records', indent=2)
                st.download_button("📥 Download JSON", json_data,
                                   f"full_data_{datetime.now().strftime('%Y%m%d')}.json",
                                   "application/json", use_container_width=True)
        else:
            st.markdown("""
            <div class="custom-alert alert-warning">
                <span>⚠️</span> No data loaded. Train the system first.
            </div>
            """, unsafe_allow_html=True)

    with tab_s3:
        st.markdown("""
        <div class="section-card" style="text-align:center;">
            <div style="font-size:3rem; margin-bottom:1rem;">🤖</div>
            <h3 style="justify-content:center;">AI Debt Collection Agents v2.0</h3>
            <p style="color:var(--text-secondary); max-width:500px; margin:0 auto;">
                A multi-agent AI platform for intelligent debt collection management.
                Featuring risk assessment, NLP communication analysis, payment prediction,
                reinforcement learning strategy optimization, and regulatory compliance checking.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        features = [
            ("🧠", "Risk Assessment Agent", "Ensemble ML models for debtor risk scoring"),
            ("💬", "NLP Communication Agent", "Sentiment analysis and intent classification"),
            ("💰", "Payment Prediction Agent", "Neural network payment probability forecasting"),
            ("🎯", "RL Strategy Agent", "Reinforcement learning for optimal collection strategies"),
            ("⚖️", "Compliance Engine", "FDCPA, TCPA, and SCRA rule validation"),
            ("🔄", "Orchestrator Agent", "Multi-agent coordination and priority scoring"),
        ]

        fc1, fc2 = st.columns(2)
        for i, (icon, name, desc) in enumerate(features):
            with fc1 if i % 2 == 0 else fc2:
                st.markdown(f"""
                <div class="analysis-panel">
                    <div style="display:flex; align-items:center; gap:0.75rem;">
                        <span style="font-size:1.5rem;">{icon}</span>
                        <div>
                            <div style="font-weight:700; color:var(--text-primary);
                                        font-size:0.9rem;">{name}</div>
                            <div style="font-size:0.78rem; color:var(--text-secondary);">{desc}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    render_footer()

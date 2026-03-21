import streamlit as st
import pandas as pd
import plotly.express as px
import time
import json
from datetime import datetime
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
# CUSTOM CSS - LIGHT THEME
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
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-card: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border: #e2e8f0;
        --gradient-1: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        --gradient-2: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        --gradient-3: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        --gradient-4: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.08);
        --shadow-lg: 0 10px 25px rgba(0,0,0,0.1);
        --shadow-glow: 0 0 20px rgba(99,102,241,0.15);
    }

    /* ── Global Styles ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }

    .main .block-container {
        padding: 1.5rem 2rem 3rem 2rem;
        max-width: 1400px;
        background-color: var(--bg-secondary);
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

    /* ── Sidebar Styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* ── Header Cards ── */
    .page-header {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }

    .page-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
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
        box-shadow: var(--shadow-sm);
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-glow);
        border-color: var(--primary-light);
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
        top: 0; left: 0; right: 0; height: 4px;
        background: var(--gradient-1);
    }
    .metric-card.blue::before {
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: var(--gradient-2);
    }
    .metric-card.green::before {
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: var(--gradient-3);
    }
    .metric-card.amber::before {
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
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
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        color: #dc2626;
        border: 1px solid rgba(239,68,68,0.3);
    }

    .risk-medium {
        background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05));
        color: #d97706;
        border: 1px solid rgba(245,158,11,0.3);
    }

    .risk-low {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
        color: #059669;
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
        background: linear-gradient(135deg, #ffffff, #f1f5f9);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: var(--shadow-md);
    }

    .profile-avatar {
        width: 72px; height: 72px;
        border-radius: 50%;
        background: var(--gradient-1);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem; font-weight: 800; color: white;
        margin: 0 auto 1rem auto;
        box-shadow: 0 4px 15px rgba(99,102,241,0.3);
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
        background: rgba(255,255,255,0.9);
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
        box-shadow: var(--shadow-sm);
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
        box-shadow: var(--shadow-sm);
    }

    .training-progress h3 {
        color: var(--primary);
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
        color: #2563eb;
    }

    .alert-success {
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.25);
        color: #059669;
    }

    .alert-warning {
        background: rgba(245,158,11,0.1);
        border: 1px solid rgba(245,158,11,0.25);
        color: #d97706;
    }

    .alert-danger {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.25);
        color: #dc2626;
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
        color: #dc2626;
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

    /* ── Scroll to Top Button ── */
    .scroll-to-top {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 50px;
        height: 50px;
        background: var(--gradient-1);
        color: white;
        border: none;
        border-radius: 50%;
        font-size: 1.5rem;
        cursor: pointer;
        display: none;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 15px rgba(99,102,241,0.4);
        z-index: 1000;
        transition: all 0.3s;
    }

    .scroll-to-top:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.5);
    }

    .scroll-to-top.show {
        display: flex;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SCROLL TO TOP FUNCTIONALITY
# =========================================================
st.markdown("""
<script>
window.addEventListener('scroll', function() {
    const scrollTopBtn = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollTopBtn.classList.add('show');
    } else {
        scrollTopBtn.classList.remove('show');
    }
});

document.querySelector('.scroll-to-top').addEventListener('click', function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
});
</script>
<button class="scroll-to-top" title="Scroll to Top">⬆️</button>
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
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'color': '#64748b'}},
        number={'font': {'size': 32, 'color': '#1e293b'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#cbd5e1',
                     'tickfont': {'color': '#94a3b8'}},
            'bar': {'color': '#6366f1'},
            'bgcolor': '#ffffff',
            'borderwidth': 1,
            'bordercolor': '#e2e8f0',
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
        font={'color': '#1e293b'},
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
# SESSION STATE INITIALIZATION - FIXED ERROR
# =========================================================
if 'nav_page' not in st.session_state:
    st.session_state['nav_page'] = "System Setup & Training"

if 'analysis_history' not in st.session_state:
    st.session_state['analysis_history'] = []

if 'batch_results' not in st.session_state:
    st.session_state['batch_results'] = None

if 'export_format' not in st.session_state:
    st.session_state['export_format'] = 'csv'

if 'redirect_to_dashboard' not in st.session_state:
    st.session_state['redirect_to_dashboard'] = False


# =========================================================
# REDIRECT HANDLER (FIXES SESSION STATE ERROR)
# =========================================================
if st.session_state['redirect_to_dashboard']:
    del st.session_state['redirect_to_dashboard']
    st.rerun()

# Check for redirect flag first
if st.session_state.get('redirect_to_dashboard', False):
    st.session_state['redirect_to_dashboard'] = False
    st.rerun()

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
        <div style="font-size:0.7rem; color:#64748b; margin-top:0.15rem;">
            Multi-Agent Collection Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
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
    status = api.get_status()

    if status['system_initialized']:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2);
                    border-radius:10px; padding:0.9rem;">
            <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.6rem;">
                <span class="badge badge-success">● ONLINE</span>
            </div>
            <div style="font-size:0.75rem; color:#64748b; line-height:1.8;">
                👥 Debtors: <strong style="color:#1e293b">{status['data_stats'].get('debtors_loaded', 0):,}</strong><br>
                💬 Communications: <strong style="color:#1e293b">{status['data_stats'].get('communications_loaded', 0):,}</strong><br>
                🕐 Last Updated: <strong style="color:#1e293b">{datetime.now().strftime('%H:%M')}</strong>
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
            <div style="font-size:0.75rem; color:#64748b;">
                Train models to activate the system.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Actions
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
                       "The AI agents need to be trained before you can use this module.")

    st.markdown("""
    <div class="custom-alert alert-warning">
        <span>⚠️</span>
        <div>
            <strong>Action Required:</strong> Navigate to
            <em>System Setup &amp; Training</em> in the sidebar.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Go to Setup", type="primary", use_container_width=True):
            st.session_state['nav_page'] = "System Setup & Training"
            st.rerun()
    st.stop()

# =========================================================
# PAGE: SYSTEM SETUP & TRAINING
# =========================================================
if page == "System Setup & Training":
    render_page_header(
        "⚙️", "System Setup & Training",
        "Initialize and train the multi-agent AI system."
    )

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
                Use the built-in data generator to simulate realistic debtor profiles.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            n_debtors = st.slider("📊 Number of Debtor Profiles", 100, 2000, 500, step=100)
        with col2:
            n_comms = st.slider("💬 Number of Communications", 500, 5000, 1000, step=100)

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

        if st.button("🚀 Initialize & Train System", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

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

                steps = [
                    ("Generating synthetic debtor profiles...", 15),
                    ("Generating communication records...", 30),
                    ("Training Risk Assessment Agent...", 50),
                    ("Training NLP & Sentiment Agent...", 65),
                    ("Training Payment Prediction Agent...", 80),
                    ("Training RL Strategy Agent...", 90),
                    ("Validating and deploying models...", 95),
                ]

                for label, pct in steps:
                    status_text.markdown(f"""
                    <div class="custom-alert alert-info">
                        <span class="pulse">⏳</span> <strong>{label}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    progress_bar.progress(pct)
                    time.sleep(max(0.5, est_time / len(steps) * 0.3))
                    if not thread.is_alive():
                        break

                thread.join()
                progress_bar.progress(100)

                res_status, res_err = result_queue.get_nowait()
                if res_status == "success":
                    status_text.markdown("""
                    <div class="custom-alert alert-success">
                        <span>✅</span> <strong>All models trained successfully!</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # FIX: Use redirect flag instead of direct assignment
                    st.session_state['redirect_to_dashboard'] = True
                else:
                    status_text.empty()
                    st.error(f"Training failed: {res_err}")

            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")

    # --- TAB 2: CUSTOM DATA ---
    with tab2:
        st.markdown("""
        <div class="section-card">
            <h3>📁 Upload Your Agency Data</h3>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            debtor_file = st.file_uploader("📊 Upload Debtors Data", type=['csv', 'xlsx'])
        with c2:
            comm_file = st.file_uploader("💬 Upload Communications Data", type=['csv', 'xlsx'])

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
                        
                        st.success("Successfully trained models on your data!")
                        
                        # FIX: Use redirect flag
                        st.session_state['redirect_to_dashboard'] = True
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please upload both files to proceed.")

    render_footer()

# =========================================================
# PAGE: DASHBOARD OVERVIEW
# =========================================================
elif page == "Dashboard Overview":
    render_page_header(
        "📊", "Agency Dashboard",
        "Real-time overview of your debt collection portfolio."
    )

    df = api.orchestrator.debtor_data

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("👥", "Total Debtors", f"{len(df):,}",
                           color_class="purple")
    with c2:
        render_metric_card("💰", "Total Outstanding", f"${df['remaining_balance'].sum():,.0f}",
                           color_class="blue")
    with c3:
        render_metric_card("📊", "Avg Credit Score", f"{int(df['credit_score'].mean())}",
                           color_class="green")
    with c4:
        render_metric_card("📞", "Avg Response Rate", f"{df['response_rate'].mean():.1%}",
                           color_class="amber")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("""
        <div class="section-card"><h3>📊 Debt Status Distribution</h3></div>
        """, unsafe_allow_html=True)
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        status_counts['Status'] = status_counts['Status'].str.replace('_', ' ').str.title()

        fig1 = px.pie(status_counts, names='Status', values='Count', hole=0.45)
        fig1.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with c_right:
        st.markdown("""
        <div class="section-card"><h3>🏷️ Debt Types Breakdown</h3></div>
        """, unsafe_allow_html=True)
        if 'debt_type' in df.columns:
            type_counts = df['debt_type'].value_counts().reset_index()
            type_counts.columns = ['Debt Type', 'Count']
            type_counts['Debt Type'] = type_counts['Debt Type'].str.replace('_', ' ').str.title()

            fig2 = px.bar(type_counts, x='Debt Type', y='Count', color_discrete_sequence=['#6366f1'])
            fig2.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    render_footer()

# =========================================================
# PAGE: DEBTOR ANALYSIS PROFILER
# =========================================================
elif page == "Debtor Analysis Profiler":
    render_page_header(
        "🔍", "Comprehensive Debtor Profiler",
        "Deep-dive analysis of individual debtors using all AI agents."
    )

    df = api.orchestrator.debtor_data
    debtor_ids = df['debtor_id'].tolist()

    col_profile, col_analysis = st.columns([1, 2.5])

    with col_profile:
        selected_id = st.selectbox("🔎 Search Debtor", debtor_ids)
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

    with col_analysis:
        custom_msg = st.text_area(
            "Debtor's recent email/call transcript (optional)",
            placeholder="Enter debtor message here...",
            height=100
        )

        run_analysis = st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)

        if run_analysis:
            with st.spinner("AI agents are analyzing..."):
                analysis = api.analyze_debtor(debtor_id=selected_id, message=custom_msg if custom_msg else None)

            if 'error' in analysis:
                st.error(analysis['error'])
            else:
                g1, g2, g3 = st.columns(3)
                with g1:
                    fig_gauge = create_gauge_chart(analysis['priority_score'], "Priority Score")
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

    render_footer()

# =========================================================
# PAGE: BATCH PRIORITY ANALYSIS
# =========================================================
elif page == "Batch Priority Analysis":
    render_page_header(
        "📑", "Batch Queue Optimization",
        "Analyze multiple debtors simultaneously to create priority-ranked collection queue."
    )

    batch_size = st.slider("📊 Debtors to Analyze", 10, 100, 25, step=5)

    if st.button("🚀 Run Batch Optimization", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {batch_size} debtors across all AI agents..."):
            results = api.batch_analyze(n=batch_size)
            
            clean_results = []
            for i, r in enumerate(results):
                if 'error' not in r:
                    clean_results.append({
                        "Debtor ID": r['debtor_id'],
                        "Priority Score": r['priority_score'],
                        "Risk Level": r['agents_results']['risk_assessment']['risk_level'].upper(),
                        "Strategy": r['agents_results']['strategy']['recommended_strategy'].replace('_', ' ').title(),
                        "Payment Prob.": f"{r['agents_results']['risk_assessment']['payment_probability']:.1%}",
                        "Compliance": "✅ Pass" if r['agents_results']['compliance']['is_approved'] else "❌ Fail"
                    })
            
            st.session_state['batch_results'] = clean_results

    if st.session_state['batch_results']:
        results_df = pd.DataFrame(st.session_state['batch_results'])
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    render_footer()

# =========================================================
# PAGE: COMPLIANCE CHECKER
# =========================================================
elif page == "Compliance Checker":
    render_page_header(
        "⚖️", "FDCPA / TCPA Compliance Sandbox",
        "Test proposed collection actions against federal regulations."
    )

    col_flags, col_action = st.columns([1, 1.5])

    with col_flags:
        st.markdown("""
        <div class="section-card"><h3>🛡️ Debtor Protection Flags</h3></div>
        """, unsafe_allow_html=True)

        dnc = st.toggle("📵 Do Not Call Registry", value=False)
        cnd = st.toggle("🛑 Cease and Desist Requested", value=False)
        att = st.toggle("⚖️ Represented by Attorney", value=False)
        bnk = st.toggle("📋 Bankruptcy Filed", value=False)
        mil = st.toggle("🎖️ Active Military (SCRA)", value=False)

    with col_action:
        st.markdown("""
        <div class="section-card"><h3>📤 Proposed Collection Action</h3></div>
        """, unsafe_allow_html=True)

        channel = st.selectbox("📱 Communication Channel", ["phone", "email", "sms", "letter"])
        contact_time = st.time_input("🕐 Time of Contact")
        msg = st.text_area("📝 Message Content", height=150)

    if st.button("🔍 Run Compliance Check", type="primary", use_container_width=True):
        if msg:
            debtor_context = {
                'do_not_call': dnc, 'cease_and_desist': cnd,
                'represented_by_attorney': att, 'bankruptcy_filed': bnk,
                'active_military': mil
            }
            action_context = {
                'channel': channel,
                'time': contact_time.strftime("%H:%M"),
                'message': msg
            }

            with st.spinner("Checking compliance..."):
                result = api.check_compliance(action_context, debtor_context)

            if result['compliance_status'] == 'compliant':
                st.success("✅ Fully Compliant - Safe to proceed")
            else:
                st.error(f"❌ Non-Compliant - {result['compliance_status'].upper()}")

    render_footer()

# =========================================================
# PAGE: ANALYTICS & REPORTS
# =========================================================
elif page == "Analytics & Reports":
    render_page_header(
        "📈", "Analytics & Reports",
        "Comprehensive portfolio analytics and performance insights."
    )

    df = api.orchestrator.debtor_data

    st.markdown("""
    <div class="section-card"><h3>📊 Portfolio Analytics</h3></div>
    """, unsafe_allow_html=True)

    fig_corr = px.scatter(df.sample(min(300, len(df))), x='days_past_due', y='remaining_balance',
                          color='credit_score', size='remaining_balance', opacity=0.7)
    st.plotly_chart(fig_corr, use_container_width=True)

    render_footer()

# =========================================================
# PAGE: SETTINGS
# =========================================================
elif page == "Settings":
    render_page_header(
        "🔧", "Settings & Configuration",
        "Customize application behavior and manage data."
    )

    if api.initialized:
        st.markdown(f"""
        <div class="custom-alert alert-info">
            <span>ℹ️</span>
            <div>
                <strong>Current Dataset:</strong> {len(api.orchestrator.debtor_data):,} debtors loaded
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Clear All Data & Reset System", type="primary"):
        st.cache_resource.clear()
        st.rerun()

    render_footer()

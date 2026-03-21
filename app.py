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
        'About': '## AI Debt Collection Agents v2.1\nPowered by Multi-Agent AI Systems'
    }
)

# =========================================================
# CUSTOM CSS - LIGHT THEME + SCROLL TO TOP
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary: #4f46e5;
        --primary-light: #6366f1;
        --primary-dark: #4338ca;
        --success: #059669;
        --success-light: #10b981;
        --warning: #d97706;
        --warning-light: #f59e0b;
        --danger: #dc2626;
        --danger-light: #ef4444;
        --info: #2563eb;
        --info-light: #3b82f6;
        --bg-primary: #f8fafc;
        --bg-secondary: #ffffff;
        --bg-card: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --border: #e2e8f0;
        --gradient-1: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        --gradient-2: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
        --gradient-3: linear-gradient(135deg, #059669 0%, #10b981 100%);
        --gradient-4: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.07);
        --shadow-lg: 0 10px 25px rgba(0,0,0,0.1);
        --shadow-glow: 0 0 20px rgba(79,70,229,0.15);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    .main .block-container {
        padding: 1.5rem 2rem 3rem 2rem;
        max-width: 1400px;
        background-color: var(--bg-primary);
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid var(--border);
    }

    .page-header {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }

    .page-header::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: var(--gradient-1);
    }

    .page-header h1 {
        font-size: 1.85rem; font-weight: 800; margin: 0 0 0.5rem 0;
        background: var(--gradient-1); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; background-clip: text;
        letter-spacing: -0.02em;
    }

    .page-header p { color: var(--text-secondary); font-size: 0.95rem; margin: 0; line-height: 1.6; }

    .metric-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 14px; padding: 1.4rem 1.6rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative; overflow: hidden; box-shadow: var(--shadow-sm);
    }

    .metric-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-glow); border-color: var(--primary); }
    .metric-card .metric-icon { font-size: 2rem; margin-bottom: 0.6rem; display: block; }
    .metric-card .metric-label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: var(--text-secondary); margin-bottom: 0.35rem; }
    .metric-card .metric-value { font-size: 1.75rem; font-weight: 800; color: var(--text-primary); line-height: 1.2; }
    .metric-card .metric-delta { font-size: 0.8rem; font-weight: 600; margin-top: 0.4rem;
        display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.15rem 0.5rem; border-radius: 6px; }
    .metric-delta.positive { color: var(--success); background: rgba(5,150,105,0.1); }
    .metric-delta.negative { color: var(--danger); background: rgba(220,38,38,0.1); }

    .metric-card.purple::before, .metric-card.blue::before,
    .metric-card.green::before, .metric-card.amber::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    }
    .metric-card.purple::before { background: var(--gradient-1); }
    .metric-card.blue::before { background: var(--gradient-2); }
    .metric-card.green::before { background: var(--gradient-3); }
    .metric-card.amber::before { background: var(--gradient-4); }

    .section-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 14px; padding: 1.5rem 1.8rem; margin-bottom: 1rem; box-shadow: var(--shadow-sm);
    }
    .section-card h3 { font-size: 1.05rem; font-weight: 700; color: var(--text-primary);
        margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }

    .badge { display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.3rem 0.75rem;
        border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em; }
    .badge-success { background: rgba(5,150,105,0.1); color: var(--success); border: 1px solid rgba(5,150,105,0.2); }
    .badge-danger { background: rgba(220,38,38,0.1); color: var(--danger); border: 1px solid rgba(220,38,38,0.2); }
    .badge-warning { background: rgba(217,119,6,0.1); color: var(--warning); border: 1px solid rgba(217,119,6,0.2); }
    .badge-info { background: rgba(37,99,235,0.1); color: var(--info); border: 1px solid rgba(37,99,235,0.2); }
    .badge-purple { background: rgba(79,70,229,0.1); color: var(--primary); border: 1px solid rgba(79,70,229,0.2); }

    .risk-indicator { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem;
        border-radius: 10px; font-weight: 700; font-size: 0.9rem; }
    .risk-high { background: rgba(220,38,38,0.08); color: var(--danger); border: 1px solid rgba(220,38,38,0.2); }
    .risk-medium { background: rgba(217,119,6,0.08); color: var(--warning); border: 1px solid rgba(217,119,6,0.2); }
    .risk-low { background: rgba(5,150,105,0.08); color: var(--success); border: 1px solid rgba(5,150,105,0.2); }

    .progress-container { display: flex; align-items: center; justify-content: space-between;
        margin: 1.5rem 0; padding: 0 1rem; }
    .progress-step { display: flex; flex-direction: column; align-items: center; gap: 0.5rem; flex: 1; }
    .step-circle { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center;
        justify-content: center; font-weight: 700; font-size: 0.9rem; transition: all 0.3s; }
    .step-circle.active { background: var(--gradient-1); color: white; box-shadow: 0 0 15px rgba(79,70,229,0.3); }
    .step-circle.completed { background: var(--success); color: white; }
    .step-circle.pending { background: #f1f5f9; color: var(--text-secondary); border: 2px solid var(--border); }
    .step-label { font-size: 0.72rem; font-weight: 600; color: var(--text-secondary); text-align: center; }
    .step-connector { flex: 1; height: 2px; background: var(--border); margin: 0 0.5rem; margin-bottom: 1.5rem; }
    .step-connector.completed { background: var(--success); }

    .profile-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px;
        padding: 1.8rem; text-align: center; box-shadow: var(--shadow-sm); }
    .profile-avatar { width: 72px; height: 72px; border-radius: 50%; background: var(--gradient-1);
        display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: 800;
        color: white; margin: 0 auto 1rem auto; box-shadow: 0 0 20px rgba(79,70,229,0.2); }
    .profile-name { font-size: 1.2rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.25rem; }
    .profile-id { font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 1rem; }
    .profile-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 1rem; }
    .profile-stat { background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: 0.75rem; }
    .profile-stat-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: var(--text-secondary); font-weight: 600; }
    .profile-stat-value { font-size: 1.05rem; font-weight: 700; color: var(--text-primary); margin-top: 0.15rem; }

    .analysis-panel { background: #f8fafc; border: 1px solid var(--border); border-radius: 14px;
        padding: 1.5rem; margin-bottom: 0.75rem; }
    .analysis-panel h4 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: var(--text-secondary); font-weight: 600; margin-bottom: 0.75rem; }

    .custom-alert { padding: 1rem 1.25rem; border-radius: 10px; font-size: 0.88rem; font-weight: 500;
        display: flex; align-items: flex-start; gap: 0.6rem; margin-bottom: 0.75rem; }
    .alert-info { background: rgba(37,99,235,0.08); border: 1px solid rgba(37,99,235,0.2); color: #1e3a8a; }
    .alert-success { background: rgba(5,150,105,0.08); border: 1px solid rgba(5,150,105,0.2); color: #064e3b; }
    .alert-warning { background: rgba(217,119,6,0.08); border: 1px solid rgba(217,119,6,0.2); color: #78350f; }
    .alert-danger { background: rgba(220,38,38,0.08); border: 1px solid rgba(220,38,38,0.2); color: #7f1d1d; }

    .violation-card { background: rgba(220,38,38,0.05); border: 1px solid rgba(220,38,38,0.15);
        border-left: 4px solid var(--danger); border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 0.6rem; }
    .violation-card .violation-rule { font-weight: 700; color: var(--danger); font-size: 0.88rem; }
    .violation-card .violation-desc { color: var(--text-secondary); font-size: 0.82rem; margin-top: 0.25rem; }
    .warning-card { background: rgba(217,119,6,0.05); border: 1px solid rgba(217,119,6,0.15);
        border-left: 4px solid var(--warning); border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 0.6rem; }

    .app-footer { text-align: center; padding: 2rem 0 1rem 0; color: var(--text-secondary);
        font-size: 0.75rem; border-top: 1px solid var(--border); margin-top: 3rem; }

    @media (max-width: 768px) {
        .main .block-container { padding: 1rem; }
        .page-header { padding: 1.5rem; }
        .page-header h1 { font-size: 1.4rem; }
        .metric-card .metric-value { font-size: 1.3rem; }
        .profile-stats { grid-template-columns: 1fr; }
    }

    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }

    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .animate-in { animation: fadeInUp 0.5s ease-out; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    .pulse { animation: pulse 2s infinite; }

    /* Scroll to Top Button */
    #scrollTopBtn {
        position: fixed; bottom: 30px; right: 30px; z-index: 999;
        border: none; outline: none; background: var(--gradient-1);
        color: white; cursor: pointer; padding: 14px 16px;
        border-radius: 50%; font-size: 20px; font-weight: bold;
        box-shadow: 0 4px 12px rgba(79,70,229,0.3);
        opacity: 0; visibility: hidden; transition: all 0.3s ease;
    }
    #scrollTopBtn.show { opacity: 1; visibility: visible; }
    #scrollTopBtn:hover { transform: translateY(-3px); box-shadow: 0 6px 16px rgba(79,70,229,0.4); }
</style>

<button id="scrollTopBtn" title="Scroll to top">↑</button>
<script>
    const btn = document.getElementById("scrollTopBtn");
    window.onscroll = function() {
        if (document.body.scrollTop > 400 || document.documentElement.scrollTop > 400) {
            btn.classList.add("show");
        } else {
            btn.classList.remove("show");
        }
    };
    btn.onclick = function() {
        window.scrollTo({ top: 0, behavior: "smooth" });
    };
</script>
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
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title, 'font': {'size': 14, 'color': '#475569'}},
        number={'font': {'size': 32, 'color': '#0f172a'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#cbd5e1', 'tickfont': {'color': '#475569'}},
            'bar': {'color': '#4f46e5'}, 'bgcolor': '#ffffff', 'borderwidth': 1, 'bordercolor': '#e2e8f0',
            'steps': [
                {'range': [0, max_val * 0.33], 'color': 'rgba(5,150,105,0.1)'},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': 'rgba(217,119,6,0.1)'},
                {'range': [max_val * 0.66, max_val], 'color': 'rgba(220,38,38,0.1)'}
            ],
            'threshold': {'line': {'color': '#6366f1', 'width': 3}, 'thickness': 0.8, 'value': value}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font={'color': '#0f172a'}, height=220, margin=dict(l=30, r=30, t=40, b=10))
    return fig

def render_footer():
    st.markdown("""
    <div class="app-footer">
        <p>🤖 AI Debt Collection Agents v2.1 — Powered by Multi-Agent AI Systems</p>
        <p>Built with ❤️ using Streamlit · FDCPA & TCPA Compliant</p>
    </div>
    """, unsafe_allow_html=True)

def light_plotly_layout(fig):
    fig.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font=dict(color='#0f172a', family='Inter'),
        xaxis=dict(gridcolor='#e2e8f0', zerolinecolor='#e2e8f0'),
        yaxis=dict(gridcolor='#e2e8f0', zerolinecolor='#e2e8f0'),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


# =========================================================
# CACHING & STATE INIT
# =========================================================
@st.cache_resource(show_spinner=False)
def load_ai_system():
    return DebtCollectionAPI()

api = load_ai_system()

if "current_page" not in st.session_state:
    st.session_state.current_page = "System Setup & Training"
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 0.5rem 0;">
        <div style="font-size:2.5rem; margin-bottom:0.25rem;">🤖</div>
        <div style="font-size:1.1rem; font-weight:800; color:#4f46e5; letter-spacing:-0.02em;">DebtCollect AI</div>
        <div style="font-size:0.7rem; color:#475569; margin-top:0.15rem;">Multi-Agent Collection Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    nav_items = {
        "System Setup & Training": "⚙️", "Dashboard Overview": "📊",
        "Debtor Analysis Profiler": "🔍", "Batch Priority Analysis": "📑",
        "Compliance Checker": "⚖️", "Analytics & Reports": "📈", "Settings": "🔧"
    }

    # SAFE NAVIGATION PATTERN: Decouple widget from direct state mutation
    page_list = list(nav_items.keys())
    selected_idx = page_list.index(st.session_state.current_page) if st.session_state.current_page in page_list else 0
    
    selected_page = st.sidebar.radio(
        "Navigate", page_list, index=selected_idx,
        format_func=lambda x: f"{nav_items[x]}  {x}", key="nav_radio", label_visibility="collapsed"
    )
    st.session_state.current_page = selected_page
    page = st.session_state.current_page

    st.markdown("---")
    st.markdown('<p style="font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#64748b; margin-bottom:0.5rem;">🖥️ System Status</p>', unsafe_allow_html=True)
    status = api.get_status()

    if status['system_initialized']:
        st.markdown(f"""
        <div style="background:rgba(5,150,105,0.08); border:1px solid rgba(5,150,105,0.2); border-radius:10px; padding:0.9rem;">
            <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.6rem;">
                <span class="badge badge-success">● ONLINE</span>
            </div>
            <div style="font-size:0.75rem; color:#475569; line-height:1.8;">
                👥 Debtors: <strong style="color:#0f172a">{status['data_stats'].get('debtors_loaded', 0):,}</strong><br>
                💬 Communications: <strong style="color:#0f172a">{status['data_stats'].get('communications_loaded', 0):,}</strong><br>
                🕐 Last Updated: <strong style="color:#0f172a">{datetime.now().strftime('%H:%M')}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(220,38,38,0.08); border:1px solid rgba(220,38,38,0.2); border-radius:10px; padding:0.9rem;">
            <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.4rem;"><span class="badge badge-danger">● OFFLINE</span></div>
            <div style="font-size:0.75rem; color:#475569;">Train models to activate the system.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#64748b; margin-bottom:0.5rem;">⚡ Quick Actions</p>', unsafe_allow_html=True)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    if api.initialized:
        df = api.orchestrator.debtor_data
        csv = df.to_csv(index=False)
        st.download_button("📥 Export All Data", csv, f"debtors_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)


# =========================================================
# GATE: require initialization
# =========================================================
if not api.initialized and page not in ("System Setup & Training", "Settings"):
    render_page_header("⚠️", "System Not Initialized", "Train the AI agents before using this module.")
    st.markdown('<div class="custom-alert alert-warning"><span>⚠️</span><div><strong>Action Required:</strong> Go to <em>System Setup & Training</em> to initialize.</div></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Go to Setup", type="primary", use_container_width=True):
            st.session_state.current_page = "System Setup & Training"
            st.rerun()
    st.stop()


# =========================================================
# PAGE: SYSTEM SETUP & TRAINING
# =========================================================
if page == "System Setup & Training":
    render_page_header("⚙️", "System Setup & Training", "Initialize and train the multi-agent AI system.")
    
    step_cls = "completed" if api.initialized else "active"
    conn_cls = "completed" if api.initialized else ""
    st.markdown(f"""
    <div class="progress-container animate-in">
        <div class="progress-step"><div class="step-circle {step_cls}">1</div><span class="step-label">Load Data</span></div>
        <div class="step-connector {conn_cls}"></div>
        <div class="progress-step"><div class="step-circle {step_cls}">2</div><span class="step-label">Train Models</span></div>
        <div class="step-connector {conn_cls}"></div>
        <div class="progress-step"><div class="step-circle {step_cls}">3</div><span class="step-label">Validate</span></div>
        <div class="step-connector {conn_cls}"></div>
        <div class="progress-step"><div class="step-circle {step_cls}">4</div><span class="step-label">Deploy</span></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🧬 Synthetic Data Generator", "📁 Upload Custom Data"])

    with tab1:
        st.markdown('<div class="section-card"><h3>🧬 Generate Synthetic Training Data</h3><p style="color:var(--text-secondary); font-size:0.88rem;">Simulate realistic debtor profiles for quick deployment.</p></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        n_debtors = c1.slider("📊 Debtor Profiles", 100, 2000, 500, step=100)
        n_comms = c2.slider("💬 Communications", 500, 5000, 1000, step=100)
        
        est_time = max(30, (n_debtors + n_comms) // 50)
        st.markdown(f'<div class="custom-alert alert-info"><span>💡</span><div>Est. training time: <strong>{est_time}s</strong>. Keep under 1,500 for free cloud tiers.</div></div>', unsafe_allow_html=True)
        
        training_successful = False
        if st.button("🚀 Initialize & Train System", type="primary", use_container_width=True):
            progress = st.progress(0)
            status_txt = st.empty()
            try:
                for i, label in enumerate(["Generating debtors...", "Generating comms...", "Training Risk Agent...", "Training NLP Agent...", "Training Payment Agent...", "Training RL Agent...", "Validating..."]):
                    status_txt.markdown(f'<div class="custom-alert alert-info"><span class="pulse">⏳</span> <strong>{label}</strong></div>', unsafe_allow_html=True)
                    progress.progress(int((i+1)/7 * 100))
                    time.sleep(0.4)
                api.setup(n_training_debtors=n_debtors, n_training_comms=n_comms)
                progress.progress(100)
                training_successful = True
                status_txt.markdown('<div class="custom-alert alert-success"><span>✅</span> <strong>Training Complete!</strong></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

        if training_successful:
            st.toast("✅ System Successfully Trained! Redirecting...", icon="🎉")
            st.session_state.current_page = "Dashboard Overview"
            st.rerun()

    with tab2:
        st.markdown('<div class="section-card"><h3>📁 Upload Your Agency Data</h3><p style="color:var(--text-secondary); font-size:0.88rem;">Train on real data for production accuracy.</p></div>', unsafe_allow_html=True)
        with st.expander("📋 Data Format Requirements"):
            st.markdown("**Debtors:** `debtor_id`, `first_name`, `last_name`, `total_debt`, `remaining_balance`, `days_past_due`, `credit_score`, `income_estimate`, `response_rate`, `status`, `will_pay_30_days` (0/1)\n\n**Comms:** `comm_id`, `debtor_id`, `text`, `intent`, `sentiment`")
        
        c1, c2 = st.columns(2)
        debtor_file = c1.file_uploader("📊 Debtors Data", type=['csv', 'xlsx'])
        comm_file = c2.file_uploader("💬 Communications Data", type=['csv', 'xlsx'])
        
        training_successful_upload = False
        if st.button("🚀 Train with Uploaded Data", type="primary", use_container_width=True):
            if debtor_file and comm_file:
                with st.spinner("Processing & training..."):
                    try:
                        df_d = pd.read_csv(debtor_file) if debtor_file.name.endswith('.csv') else pd.read_excel(debtor_file)
                        df_c = pd.read_csv(comm_file) if comm_file.name.endswith('.csv') else pd.read_excel(comm_file)
                        orig_d = api.orchestrator.data_generator.generate_debtor_profiles
                        orig_c = api.orchestrator.data_generator.generate_communication_data
                        api.orchestrator.data_generator.generate_debtor_profiles = lambda n: df_d
                        api.orchestrator.data_generator.generate_communication_data = lambda n: df_c
                        api.setup(n_training_debtors=len(df_d), n_training_comms=len(df_c))
                        api.orchestrator.data_generator.generate_debtor_profiles = orig_d
                        api.orchestrator.data_generator.generate_communication_data = orig_c
                        training_successful_upload = True
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Upload both files.")
                
        if training_successful_upload:
            st.toast("✅ Models trained on custom data! Redirecting...", icon="🎉")
            st.session_state.current_page = "Dashboard Overview"
            st.rerun()
    render_footer()


# =========================================================
# PAGE: DASHBOARD
# =========================================================
elif page == "Dashboard Overview":
    render_page_header("📊", "Agency Dashboard", "Real-time portfolio performance and key metrics.")
    df = api.orchestrator.debtor_data
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("👥", "Total Debtors", f"{len(df):,}", color_class="purple")
    with c2: render_metric_card("💰", "Outstanding", f"${df['remaining_balance'].sum():,.0f}", color_class="blue")
    with c3: render_metric_card("📊", "Avg Credit Score", f"{int(df['credit_score'].mean())}", color_class="green")
    with c4: render_metric_card("📞", "Avg Response Rate", f"{df['response_rate'].mean():.1%}", color_class="amber")
    
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        st.markdown('<div class="section-card"><h3>📊 Debt Status</h3></div>', unsafe_allow_html=True)
        sc = df['status'].value_counts().reset_index()
        sc.columns = ['Status', 'Count']
        fig1 = px.pie(sc, names='Status', values='Count', hole=0.45, color_discrete_sequence=['#4f46e5','#2563eb','#059669','#d97706','#dc2626'])
        st.plotly_chart(light_plotly_layout(fig1), use_container_width=True)
    with cr:
        st.markdown('<div class="section-card"><h3>🏷️ Debt Types</h3></div>', unsafe_allow_html=True)
        if 'debt_type' in df.columns:
            tc = df['debt_type'].value_counts().reset_index()
            tc.columns = ['Type', 'Count']
            fig2 = px.bar(tc, x='Type', y='Count', color='Count', color_continuous_scale='Blues')
            st.plotly_chart(light_plotly_layout(fig2), use_container_width=True)
    render_footer()


# =========================================================
# PAGE: DEBTOR PROFILER
# =========================================================
elif page == "Debtor Analysis Profiler":
    render_page_header("🔍", "Comprehensive Debtor Profiler", "Deep-dive AI analysis per debtor.")
    df = api.orchestrator.debtor_data
    debtor_ids = df['debtor_id'].tolist()
    
    cp, ca = st.columns([1, 2.5])
    with cp:
        sel_id = st.selectbox("🔎 Search Debtor", debtor_ids)
        row = df[df['debtor_id'] == sel_id].iloc[0]
        fn, ln = row.get('first_name', 'U'), row.get('last_name', 'K')
        init = f"{fn[0]}{ln[0] if ln else ''}"
        st.markdown(f"""
        <div class="profile-card animate-in">
            <div class="profile-avatar">{init.upper()}</div>
            <div class="profile-name">{fn} {ln}</div>
            <div class="profile-id">ID: {sel_id}</div>
            <div style="margin-bottom:0.75rem;">{render_badge(str(row.get('status','')).replace('_',' ').title(), 'purple')}</div>
            <div class="profile-stats">
                <div class="profile-stat"><div class="profile-stat-label">Balance</div><div class="profile-stat-value">${row.get('remaining_balance',0):,.0f}</div></div>
                <div class="profile-stat"><div class="profile-stat-label">DPD</div><div class="profile-stat-value">{row.get('days_past_due',0)}</div></div>
                <div class="profile-stat"><div class="profile-stat-label">Credit</div><div class="profile-stat-value">{row.get('credit_score','N/A')}</div></div>
                <div class="profile-stat"><div class="profile-stat-label">Response</div><div class="profile-stat-value">{row.get('response_rate',0):.0%}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with ca:
        st.markdown('<div class="section-card"><h3>💬 Simulate Communication</h3></div>', unsafe_allow_html=True)
        msg = st.text_area("Transcript (optional)", placeholder="e.g. I lost my job...", height=100, label_visibility="collapsed")
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("AI agents analyzing..."):
                res = api.analyze_debtor(debtor_id=sel_id, message=msg if msg else None)
            if 'error' in res:
                st.error(res['error'])
            else:
                st.session_state.analysis_history.append({'time': datetime.now().strftime('%H:%M'), 'id': sel_id, 'score': res['priority_score']})
                g1, g2, g3 = st.columns(3)
                with g1: st.plotly_chart(light_plotly_layout(create_gauge_chart(res['priority_score'], "Priority")), use_container_width=True)
                with g2: st.plotly_chart(light_plotly_layout(create_gauge_chart(res['agents_results']['payment_prediction']['will_pay_probability']*100, "Pay Prob.")), use_container_width=True)
                with g3: 
                    rl = res['agents_results']['risk_assessment']['risk_level']
                    rv = {'low':25,'medium':55,'high':85}.get(rl.lower(),50)
                    st.plotly_chart(light_plotly_layout(create_gauge_chart(rv, "Risk Level")), use_container_width=True)
                
                t1, t2, t3, t4 = st.tabs(["⚠️ Risk/Payment", "🎯 Strategy", "💬 NLP", "⚖️ Compliance"])
                with t1:
                    r = res['agents_results']['risk_assessment']
                    p = res['agents_results']['payment_prediction']
                    st.markdown(render_risk_indicator(r['risk_level']), unsafe_allow_html=True)
                    st.metric("Payment Prob", f"{p['will_pay_probability']:.1%}")
                    st.metric("Expected Recovery", f"${p.get('expected_value',0):,.2f}")
                with t2:
                    s = res['agents_results']['strategy']
                    st.info(s['recommended_strategy'].replace('_',' ').title())
                    st.write(f"**Channel:** {s['recommended_channel']['primary'].title()}")
                    st.write(f"**Timing:** {s['recommended_timing']['best_day']} @ {s['recommended_timing']['best_time']}")
                with t3:
                    if msg:
                        c = res['agents_results'].get('communication', {})
                        st.write(f"**Sentiment:** {c.get('vader_category','N/A').title()}")
                        st.text_area("AI Draft", res['agents_results'].get('generated_response',{}).get('response',''), height=150)
                    else:
                        st.info("Add transcript for NLP analysis.")
                with t4:
                    comp = res['agents_results']['compliance']
                    if comp['is_approved']:
                        st.success(f"✅ {comp['recommendation']}")
                    else:
                        st.error(f"❌ {comp['recommendation']}")
                        for v in comp.get('violations',[]): st.warning(f"**{v['rule']}**: {v['description']}")
    render_footer()


# =========================================================
# PAGE: BATCH ANALYSIS
# =========================================================
elif page == "Batch Priority Analysis":
    render_page_header("📑", "Batch Queue Optimization", "Prioritize collection efforts at scale.")
    bs = st.slider("📊 Debtors to Analyze", 10, 100, 25, step=5)
    if st.button("🚀 Run Batch", type="primary"):
        with st.spinner("Analyzing..."):
            raw = api.batch_analyze(n=bs)
            clean = [{
                "ID": r['debtor_id'], "Priority": r['priority_score'],
                "Risk": r['agents_results']['risk_assessment']['risk_level'].upper(),
                "Strategy": r['agents_results']['strategy']['recommended_strategy'].replace('_',' ').title(),
                "Pay Prob": f"{r['agents_results']['risk_assessment']['payment_probability']:.1%}",
                "Compliance": "✅" if r['agents_results']['compliance']['is_approved'] else "❌"
            } for r in raw if 'error' not in r]
            st.session_state.batch_results = pd.DataFrame(clean)
            
    if st.session_state.batch_results is not None:
        st.dataframe(st.session_state.batch_results.sort_values('Priority', ascending=False), use_container_width=True, hide_index=True)
    render_footer()


# =========================================================
# PAGE: COMPLIANCE
# =========================================================
elif page == "Compliance Checker":
    render_page_header("⚖️", "FDCPA / TCPA Compliance Sandbox", "Test actions against federal regulations.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-card"><h3>🛡️ Debtor Flags</h3></div>', unsafe_allow_html=True)
        dnc = st.toggle("📵 Do Not Call")
        cnd = st.toggle("🛑 Cease & Desist")
        att = st.toggle("⚖️ Attorney Rep")
        bnk = st.toggle("📋 Bankruptcy")
        mil = st.toggle("🎖️ Active Military")
    with c2:
        st.markdown('<div class="section-card"><h3>📤 Proposed Action</h3></div>', unsafe_allow_html=True)
        ch = st.selectbox("Channel", ["phone","email","sms","letter"])
        tm = st.time_input("Time", value=datetime.now().replace(hour=10, minute=0))
        txt = st.text_area("Message", placeholder="Enter message...", height=120)
        
    if st.button("🔍 Check Compliance", type="primary", use_container_width=True):
        ctx_d = {'do_not_call':dnc,'cease_and_desist':cnd,'represented_by_attorney':att,'bankruptcy_filed':bnk,'active_military':mil,'contacts_today':0,'contacts_this_week':0}
        ctx_a = {'channel':ch,'time':tm.strftime("%H:%M"),'message':txt}
        res = api.check_compliance(ctx_a, ctx_d)
        if res['compliance_status'] == 'compliant':
            st.success("✅ Fully Compliant. Safe to proceed.")
        else:
            st.error(f"❌ {res['compliance_status'].upper()}")
            st.write(res.get('recommendation',''))
            for v in res.get('violations',[]): st.warning(f"**{v['rule']}**: {v['description']}")
    render_footer()


# =========================================================
# PAGE: ANALYTICS
# =========================================================
elif page == "Analytics & Reports":
    render_page_header("📈", "Analytics & Reports", "Portfolio insights and trend analysis.")
    df = api.orchestrator.debtor_data
    st.markdown('<div class="section-card"><h3>📊 Balance vs Days Past Due</h3></div>', unsafe_allow_html=True)
    fig = px.scatter(df.sample(min(300, len(df))), x='days_past_due', y='remaining_balance', color='credit_score', size='remaining_balance', opacity=0.7)
    st.plotly_chart(light_plotly_layout(fig), use_container_width=True)
    render_footer()


# =========================================================
# PAGE: SETTINGS
# =========================================================
elif page == "Settings":
    render_page_header("🔧", "Settings & Configuration", "Manage preferences and data.")
    st.markdown('<div class="section-card"><h3>⚙️ System Controls</h3></div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Cache & Reset", type="primary"):
        st.cache_resource.clear()
        st.session_state.analysis_history = []
        st.session_state.batch_results = None
        st.rerun()
    st.markdown("---")
    st.markdown("**App Version:** 2.1.0 | **Theme:** Light Mode | **Compliance:** FDCPA/TCPA/SCRA")
    render_footer()

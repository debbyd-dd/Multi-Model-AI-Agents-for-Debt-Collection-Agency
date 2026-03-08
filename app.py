import streamlit as st
import pandas as pd
import plotly.express as px
from ai_core import DebtCollectionAPI

# Page Configuration
st.set_page_config(
    page_title="AI Debt Collection Agents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CACHING THE AI SYSTEM
# ---------------------------------------------------------
# @st.cache_resource ensures the models only train once and persist across user sessions.
@st.cache_resource(show_spinner=False)
def load_ai_system():
    api = DebtCollectionAPI()
    return api

api = load_ai_system()

# ---------------------------------------------------------
# SYSTEM INITIALIZATION BARRIER
# ---------------------------------------------------------
if not api.initialized:
    st.title("🤖 Multi-Model AI Agents for Debt Collection")
    st.info("The AI system is offline. The Multi-Agent models (Risk, NLP, Payment Prediction, RL Strategy) need to be trained on synthetic data before starting.")
    
    # We reduce the default training size slightly so it doesn't timeout on Streamlit Cloud
    n_debtors = st.slider("Training Dataset Size (Debtors)", 100, 2000, 500, step=100)
    n_comms = st.slider("Training Dataset Size (Communications)", 500, 5000, 1000, step=100)
    
    if st.button("Initialize & Train System", type="primary", use_container_width=True):
        with st.spinner("Training Neural Networks, Ensembles, and RL Agents... This may take 1-2 minutes."):
            try:
                api.setup(n_training_debtors=n_debtors, n_training_comms=n_comms)
                st.success("System Successfully Trained and Initialized!")
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")
    st.stop() # Halt rendering the rest of the app until initialized

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module", 
    ["Dashboard Overview", "Debtor Analysis Profiler", "Batch Priority Analysis", "Compliance Checker"]
)

st.sidebar.divider()
st.sidebar.subheader("System Status")
status = api.get_status()
st.sidebar.caption(f"✅ System Initialized")
st.sidebar.caption(f"👥 Debtors in Database: {status['data_stats']['debtors_loaded']}")
st.sidebar.caption(f"💬 Comms in Database: {status['data_stats']['communications_loaded']}")

# ---------------------------------------------------------
# PAGE 1: DASHBOARD
# ---------------------------------------------------------
if page == "Dashboard Overview":
    st.title("📊 Agency Dashboard")
    df = api.orchestrator.debtor_data
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Debtors", f"{len(df):,}")
    col2.metric("Total Outstanding Debt", f"${df['remaining_balance'].sum():,.2f}")
    col3.metric("Avg Credit Score", f"{int(df['credit_score'].mean())}")
    col4.metric("Avg Response Rate", f"{df['response_rate'].mean():.1%}")

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Debt Status Distribution")
        fig1 = px.pie(df, names='status', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        st.subheader("Debt Types")
        fig2 = px.bar(df['debt_type'].value_counts().reset_index(), x='debt_type', y='count', color='debt_type')
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: DEBTOR ANALYSIS
# ---------------------------------------------------------
elif page == "Debtor Analysis Profiler":
    st.title("🔍 Comprehensive Debtor Profiler")
    
    df = api.orchestrator.debtor_data
    debtor_ids = df['debtor_id'].tolist()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_id = st.selectbox("Select Debtor ID to Analyze", debtor_ids)
        debtor_row = df[df['debtor_id'] == selected_id].iloc[0]
        
        st.subheader("Debtor Profile")
        st.write(f"**Name:** {debtor_row['first_name']} {debtor_row['last_name']}")
        st.write(f"**Balance:** ${debtor_row['remaining_balance']:,.2f}")
        st.write(f"**Days Past Due:** {debtor_row['days_past_due']}")
        st.write(f"**Credit Score:** {debtor_row['credit_score']}")
        st.write(f"**Status:** {debtor_row['status'].replace('_', ' ').title()}")
        
    with col2:
        st.subheader("Simulate Incoming Communication")
        custom_msg = st.text_area("Debtor's recent email/call transcript (Optional)", 
                                  placeholder="e.g. I lost my job and cannot afford this right now. Please give me some time.")
        
        if st.button("Run Multi-Agent Analysis", type="primary"):
            with st.spinner("Agents are analyzing..."):
                analysis = api.analyze_debtor(debtor_id=selected_id, message=custom_msg if custom_msg else None)
                
            if 'error' in analysis:
                st.error(analysis['error'])
            else:
                st.success(f"Analysis Complete. Priority Score: **{analysis['priority_score']}/100**")
                
                t1, t2, t3, t4 = st.tabs(["Risk & Payment", "Strategy", "Communication", "Compliance"])
                
                with t1:
                    risk = analysis['agents_results']['risk_assessment']
                    pay = analysis['agents_results']['payment_prediction']
                    st.metric("Risk Level", risk['risk_level'].upper())
                    st.metric("Probability of Payment", f"{pay['will_pay_probability']:.1%}")
                    st.metric("Expected Recovery Amount", f"${pay.get('expected_value', 0):,.2f}")
                    
                with t2:
                    strat = analysis['agents_results']['strategy']
                    st.subheader("Recommended Action")
                    st.info(strat['recommended_strategy'].replace('_', ' ').title())
                    st.write("**Best Channel:**", strat['recommended_channel']['primary'].title())
                    st.write("**Optimal Timing:**", f"{strat['recommended_timing']['best_day']} at {strat['recommended_timing']['best_time']}")
                    
                    st.write("**Alternative Strategies (Q-Values):**")
                    st.dataframe(pd.DataFrame(strat['strategy_ranking']))

                with t3:
                    if custom_msg:
                        comm = analysis['agents_results'].get('communication', {})
                        st.write("**Sentiment:**", comm.get('vader_category', 'N/A').title())
                        st.write("**Intent:**", comm.get('intent', {}).get('classification', 'N/A').replace('_', ' ').title())
                        
                        flags = comm.get('compliance_flags', [])
                        if flags:
                            st.error(f"Detected Regulatory Flags: {', '.join(flags)}")
                            
                        gen_resp = analysis['agents_results'].get('generated_response', {}).get('response', '')
                        st.text_area("AI Generated Draft Response", gen_resp, height=200)
                    else:
                        st.info("No communication text provided for NLP analysis.")
                        
                with t4:
                    comp = analysis['agents_results']['compliance']
                    if comp['is_approved']:
                        st.success("✅ " + comp['recommendation'])
                    else:
                        st.error("❌ " + comp['recommendation'])
                        for v in comp['violations']:
                            st.warning(f"**{v['rule']}**: {v['description']}")

# ---------------------------------------------------------
# PAGE 3: BATCH ANALYSIS
# ---------------------------------------------------------
elif page == "Batch Priority Analysis":
    st.title("📑 Batch Queue Optimization")
    st.write("Analyze a batch of debtors to prioritize collection efforts using the Orchestrator Agent.")
    
    batch_size = st.slider("Number of Debtors to sample", 10, 100, 20)
    
    if st.button("Run Batch Optimization"):
        with st.spinner(f"Analyzing {batch_size} debtors across all models..."):
            results = api.batch_analyze(n=batch_size)
            
            clean_results = []
            for r in results:
                if 'error' not in r:
                    clean_results.append({
                        "Debtor ID": r['debtor_id'],
                        "Priority Score": r['priority_score'],
                        "Risk Level": r['agents_results']['risk_assessment']['risk_level'].upper(),
                        "Best Strategy": r['agents_results']['strategy']['recommended_strategy'].replace('_', ' ').title(),
                        "Payment Prob.": f"{r['agents_results']['risk_assessment']['payment_probability']:.1%}",
                        "Compliance": "✅ Pass" if r['agents_results']['compliance']['is_approved'] else "❌ Fail"
                    })
            
            st.dataframe(pd.DataFrame(clean_results), use_container_width=True)

# ---------------------------------------------------------
# PAGE 4: COMPLIANCE CHECKER
# ---------------------------------------------------------
elif page == "Compliance Checker":
    st.title("⚖️ FDCPA / TCPA Compliance Sandbox")
    st.write("Manually test an intended collection action against the rules engine.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Simulated Debtor Flags")
        dnc = st.checkbox("Do Not Call Registry")
        cnd = st.checkbox("Cease and Desist Requested")
        att = st.checkbox("Represented by Attorney")
        bnk = st.checkbox("Bankruptcy Filed")
        mil = st.checkbox("Active Military (SCRA)")
        
    with c2:
        st.subheader("Proposed Action")
        channel = st.selectbox("Channel", ["phone", "email", "sms", "letter"])
        time = st.time_input("Time of Contact")
        msg = st.text_area("Message Content", "Pay your debt now or we will seize your property!")
        
    if st.button("Run Compliance Check"):
        debtor_context = {
            'do_not_call': dnc, 'cease_and_desist': cnd, 'represented_by_attorney': att,
            'bankruptcy_filed': bnk, 'active_military': mil, 'contacts_today': 0, 'contacts_this_week': 0
        }
        action_context = {'channel': channel, 'time': time.strftime("%H:%M"), 'message': msg}
        
        result = api.check_compliance(action_context, debtor_context)
        
        if result['compliance_status'] == 'compliant':
            st.success("✅ Compliant: No violations found.")
        else:
            st.error(f"❌ Status: {result['compliance_status'].upper()}")
            st.write(result['recommendation'])
            
            if 'violations' in result:
                for v in result['violations']:
                    st.error(f"**Violation ({v['severity']}) - {v['rule']}**: {v['description']}")
            if 'warnings' in result:
                for w in result['warnings']:
                    st.warning(f"**Warning**: {w}")

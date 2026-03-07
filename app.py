"""
Streamlit UI for Debt Collection AI System
This is the entry point for share.streamlit.io deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import logging

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Debt Collection AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# Import everything from main.py
# ─────────────────────────────────────────────────────────
from main import (
    DebtCollectionAPI,
    DebtCollectionDataGenerator,
    DataPreprocessor,
    RiskLevel,
    DebtStatus,
    CollectionStrategy,
    SentimentCategory,
    ComplianceStatus,
    CommunicationChannel
)


# ─────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────
def initialize_system():
    """Initialize the AI system once and cache in session state"""
    if 'api' not in st.session_state:
        with st.spinner("🚀 Initializing AI System... Training all 7 agents. This takes about 60-90 seconds on first load."):
            api = DebtCollectionAPI()
            results = api.setup(n_training_debtors=500, n_training_comms=1000)
            st.session_state.api = api
            st.session_state.setup_results = results
            st.session_state.initialized = True
    return st.session_state.api


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
        st.title("Debt Collection AI")
        st.caption("Multi-Model Agent System v2.0")

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "🏠 Dashboard",
                "🎯 Risk Scoring",
                "💬 Communication Analysis",
                "📊 Batch Analysis",
                "⚖️ Compliance Checker",
                "🔍 Debtor Lookup",
                "📈 System Status",
            ],
            index=0
        )

        st.divider()

        if st.session_state.get('initialized', False):
            st.success("✅ System Online")
            results = st.session_state.get('setup_results', {})
            train_time = results.get('total_training_time_seconds', 0)
            st.metric("Training Time", f"{train_time:.1f}s")
        else:
            st.warning("⏳ System Loading...")

        st.divider()
        st.caption("Agents: Risk Scoring • Communication • Payment Prediction • Strategy • Compliance • Segmentation • Orchestrator")

    return page


# ─────────────────────────────────────────────────────────
# Page: Dashboard
# ─────────────────────────────────────────────────────────
def page_dashboard(api: DebtCollectionAPI):
    st.title("🏠 AI System Dashboard")
    st.markdown("---")

    results = st.session_state.get('setup_results', {})

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    debtor_count = len(api.orchestrator.debtor_data) if hasattr(api.orchestrator, 'debtor_data') else 0
    comm_count = len(api.orchestrator.comm_data) if hasattr(api.orchestrator, 'comm_data') else 0

    with col1:
        st.metric("Total Debtors", f"{debtor_count:,}")
    with col2:
        st.metric("Communications", f"{comm_count:,}")
    with col3:
        risk_res = results.get('risk_agent', {})
        auc = risk_res.get('ensemble', {}).get('roc_auc', 0)
        st.metric("Risk Model AUC", f"{auc:.4f}")
    with col4:
        comm_res = results.get('communication_agent', {})
        intent_acc = comm_res.get('intent_classification', {}).get('accuracy', 0)
        st.metric("Intent Accuracy", f"{intent_acc:.4f}")

    st.markdown("---")

    # Training Results
    st.subheader("📊 Model Training Results")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Risk Scoring Agent")
        if 'risk_agent' in results:
            risk_data = results['risk_agent']
            rows = []
            for model_name, metrics in risk_data.items():
                if isinstance(metrics, dict) and 'roc_auc' in metrics:
                    rows.append({
                        'Model': model_name,
                        'AUC': metrics.get('roc_auc', 0),
                        'F1': metrics.get('f1_score', 0),
                        'Accuracy': metrics.get('accuracy', 0)
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("#### Payment Prediction Agent")
        if 'payment_agent' in results:
            pay_data = results['payment_agent']
            if 'classifier' in pay_data:
                clf = pay_data['classifier']
                st.write(f"- **AUC**: {clf.get('roc_auc', 0):.4f}")
                st.write(f"- **F1 Score**: {clf.get('f1_score', 0):.4f}")
                st.write(f"- **Epochs Trained**: {clf.get('epochs_trained', 0)}")

    with col_right:
        st.markdown("#### Communication Agent")
        if 'communication_agent' in results:
            comm_data = results['communication_agent']
            if 'intent_classification' in comm_data:
                st.write(f"- **Intent Accuracy**: {comm_data['intent_classification']['accuracy']:.4f}")
            if 'deep_sentiment' in comm_data:
                st.write(f"- **Sentiment Accuracy**: {comm_data['deep_sentiment']['accuracy']:.4f}")
                st.write(f"- **Epochs Trained**: {comm_data['deep_sentiment'].get('epochs_trained', 0)}")

        st.markdown("#### Strategy Agent")
        if 'strategy_agent' in results:
            strat_data = results['strategy_agent']
            st.write(f"- **Unique States**: {strat_data.get('unique_states', 'N/A')}")
            st.write(f"- **Avg Reward**: {strat_data.get('avg_reward_last_1000', 'N/A')}")

    st.markdown("---")

    # Data Distribution
    st.subheader("📈 Data Overview")
    if hasattr(api.orchestrator, 'debtor_data'):
        df = api.orchestrator.debtor_data

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Debt Status Distribution**")
            status_counts = df['status'].value_counts()
            st.bar_chart(status_counts)

        with col_b:
            st.markdown("**Debt Type Distribution**")
            type_counts = df['debt_type'].value_counts()
            st.bar_chart(type_counts)

        with col_c:
            st.markdown("**Payment Rate**")
            pay_rate = df['will_pay_30_days'].mean()
            st.metric("Will Pay (30 days)", f"{pay_rate:.1%}")

            avg_balance = df['remaining_balance'].mean()
            st.metric("Avg Balance", f"${avg_balance:,.0f}")

            avg_dpd = df['days_past_due'].mean()
            st.metric("Avg Days Past Due", f"{avg_dpd:.0f}")


# ─────────────────────────────────────────────────────────
# Page: Risk Scoring
# ─────────────────────────────────────────────────────────
def page_risk_scoring(api: DebtCollectionAPI):
    st.title("🎯 Risk Scoring Agent")
    st.markdown("Predict payment probability and assign risk levels using ensemble ML.")
    st.markdown("---")

    df = api.orchestrator.debtor_data

    st.subheader("Select a Debtor")
    debtor_ids = df['debtor_id'].tolist()
    selected_id = st.selectbox("Debtor ID", debtor_ids[:100], index=0)

    if st.button("🎯 Run Risk Assessment", type="primary"):
        mask = df['debtor_id'] == selected_id
        if mask.any():
            idx = mask.values.argmax()
            row = df.iloc[idx]
            feature_vector = api.orchestrator.X[idx:idx+1]

            with st.spinner("Running risk assessment..."):
                result = api.orchestrator.risk_agent.predict(feature_vector)

            if result.success:
                col1, col2, col3, col4 = st.columns(4)

                risk_level = result.data['risk_levels'][0]
                pay_prob = result.data['payment_probabilities'][0]
                risk_score = result.data['risk_scores'][0]
                confidence = result.data['confidence_scores'][0]

                color_map = {
                    'very_low': '🟢', 'low': '🟡',
                    'medium': '🟠', 'high': '🔴', 'very_high': '⛔'
                }

                with col1:
                    st.metric("Risk Level", f"{color_map.get(risk_level, '❓')} {risk_level.upper()}")
                with col2:
                    st.metric("Payment Probability", f"{pay_prob:.1%}")
                with col3:
                    st.metric("Risk Score", f"{risk_score:.4f}")
                with col4:
                    st.metric("Confidence", f"{confidence:.1%}")

                st.markdown("---")

                st.subheader("Debtor Profile")
                profile_cols = [
                    'first_name', 'last_name', 'total_debt', 'remaining_balance',
                    'days_past_due', 'credit_score', 'income_estimate',
                    'debt_type', 'employment_status', 'status',
                    'num_contact_attempts', 'response_rate'
                ]
                available = [c for c in profile_cols if c in row.index]
                profile_data = row[available].to_dict()

                col_l, col_r = st.columns(2)
                items = list(profile_data.items())
                mid = len(items) // 2
                with col_l:
                    for k, v in items[:mid]:
                        if isinstance(v, float):
                            st.write(f"**{k}**: {v:,.2f}")
                        else:
                            st.write(f"**{k}**: {v}")
                with col_r:
                    for k, v in items[mid:]:
                        if isinstance(v, float):
                            st.write(f"**{k}**: {v:,.2f}")
                        else:
                            st.write(f"**{k}**: {v}")

                # Model agreement
                st.markdown("---")
                st.subheader("Model Predictions")
                model_preds = result.data.get('model_predictions', {})
                if model_preds:
                    model_df = pd.DataFrame({
                        'Model': list(model_preds.keys()),
                        'Payment Probability': [v[0] for v in model_preds.values()]
                    })
                    st.dataframe(model_df, use_container_width=True, hide_index=True)

    # Feature importance
    st.markdown("---")
    st.subheader("📊 Feature Importance (Top 15)")
    importance = api.orchestrator.risk_agent.feature_importance
    if importance:
        top_15 = dict(list(importance.items())[:15])
        imp_df = pd.DataFrame({
            'Feature': list(top_15.keys()),
            'Importance': list(top_15.values())
        }).sort_values('Importance', ascending=True)
        st.bar_chart(imp_df.set_index('Feature'))


# ─────────────────────────────────────────────────────────
# Page: Communication Analysis
# ─────────────────────────────────────────────────────────
def page_communication(api: DebtCollectionAPI):
    st.title("💬 Communication Analysis Agent")
    st.markdown("NLP-powered sentiment analysis, intent classification, and response generation.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📝 Analyze Message", "📨 Generate Response"])

    with tab1:
        st.subheader("Analyze Debtor Communication")

        preset_messages = {
            "Custom (type your own)": "",
            "Cooperative - wants to pay": "I just got a new job and I'm ready to start paying this off. Can we set up a monthly plan?",
            "Hostile - legal threats": "I'm recording this call. You have called me 5 times today. I'm filing a complaint with the CFPB and getting a lawyer.",
            "Distressed - hardship": "I lost my job three months ago and my wife is in the hospital. I want to pay but I literally cannot afford anything right now.",
            "Settlement inquiry": "What's the lowest amount you'll accept to settle this today? I have $2,000 available right now.",
            "Dispute": "This is not my debt. I've never had an account with this creditor. I'm disputing this and I want verification.",
            "Cease and desist": "Stop calling me. I'm sending you a cease and desist letter. Do not contact me again by phone or any other means.",
        }

        selected_preset = st.selectbox("Select preset or type custom", list(preset_messages.keys()))

        if selected_preset == "Custom (type your own)":
            text_input = st.text_area("Enter debtor message:", height=120, placeholder="Type a message to analyze...")
        else:
            text_input = st.text_area("Message:", value=preset_messages[selected_preset], height=120)

        if st.button("🔍 Analyze Message", type="primary") and text_input:
            with st.spinner("Analyzing with NLP agents..."):
                result = api.analyze_message(text_input)

            if result and 'error' not in result:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### VADER Sentiment")
                    vader = result.get('vader_sentiment', {})
                    category = result.get('vader_category', 'unknown')
                    compound = vader.get('compound', 0)

                    sentiment_emoji = {
                        'very_positive': '😊', 'positive': '🙂',
                        'neutral': '😐', 'negative': '😟',
                        'very_negative': '😡'
                    }
                    st.metric("Category", f"{sentiment_emoji.get(category, '❓')} {category}")
                    st.metric("Compound Score", f"{compound:.3f}")
                    st.write(f"Positive: {vader.get('pos', 0):.3f}")
                    st.write(f"Neutral: {vader.get('neu', 0):.3f}")
                    st.write(f"Negative: {vader.get('neg', 0):.3f}")

                with col2:
                    st.markdown("#### Deep Learning Sentiment")
                    deep = result.get('deep_sentiment', {})
                    if deep:
                        st.metric("Category", deep.get('category', 'N/A'))
                        st.metric("Confidence", f"{deep.get('confidence', 0):.1%}")

                        probs = deep.get('all_probabilities', {})
                        if probs:
                            st.markdown("**All Probabilities:**")
                            prob_df = pd.DataFrame({
                                'Sentiment': list(probs.keys()),
                                'Probability': list(probs.values())
                            }).sort_values('Probability', ascending=False)
                            st.dataframe(prob_df, use_container_width=True, hide_index=True)

                with col3:
                    st.markdown("#### Intent Classification")
                    intent = result.get('intent', {})
                    if intent:
                        st.metric("Intent", intent.get('classification', 'N/A'))
                        st.metric("Confidence", f"{intent.get('confidence', 0):.1%}")

                        probs = intent.get('all_probabilities', {})
                        if probs:
                            st.markdown("**All Probabilities:**")
                            intent_df = pd.DataFrame({
                                'Intent': list(probs.keys()),
                                'Probability': list(probs.values())
                            }).sort_values('Probability', ascending=False)
                            st.dataframe(intent_df, use_container_width=True, hide_index=True)

                st.markdown("---")

                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("#### ⚠️ Risk Indicators")
                    risk_ind = result.get('risk_indicators', {})
                    for indicator, active in risk_ind.items():
                        emoji = "🔴" if active else "🟢"
                        label = indicator.replace('_', ' ').title()
                        st.write(f"{emoji} {label}")

                with col_right:
                    st.markdown("#### 🚨 Compliance Flags")
                    flags = result.get('compliance_flags', [])
                    if flags:
                        for flag in flags:
                            st.error(f"⚠️ {flag}")
                    else:
                        st.success("✅ No compliance flags detected")

                    st.markdown("#### 🔑 Keywords")
                    keywords = result.get('keywords', {})
                    for category, words in keywords.items():
                        if words:
                            label = category.replace('_', ' ').title()
                            st.write(f"**{label}**: {', '.join(words)}")

    with tab2:
        st.subheader("Generate AI Response")

        col_a, col_b = st.columns(2)
        with col_a:
            intent_choice = st.selectbox("Debtor Intent", [
                "willing_to_pay", "reluctant", "refuses_to_pay",
                "hardship", "settlement_request", "dispute"
            ])
        with col_b:
            channel_choice = st.selectbox("Channel", ["email", "phone", "sms"])

        st.markdown("**Debtor Profile for Personalization:**")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            first_name = st.text_input("First Name", "John")
        with col_p2:
            remaining = st.number_input("Remaining Balance ($)", value=5000.0, step=100.0)
        with col_p3:
            acct_num = st.text_input("Account Number", "ACC-12345")

        if st.button("📨 Generate Response", type="primary"):
            profile = {
                'first_name': first_name,
                'remaining_balance': remaining,
                'account_number': acct_num,
            }
            response = api.orchestrator.communication_agent.generate_response(
                intent=intent_choice,
                channel=channel_choice,
                debtor_profile=profile
            )
            st.markdown("#### Generated Response:")
            st.text_area("Response", value=response, height=300, disabled=True)


# ─────────────────────────────────────────────────────────
# Page: Batch Analysis
# ─────────────────────────────────────────────────────────
def page_batch_analysis(api: DebtCollectionAPI):
    st.title("📊 Batch Analysis")
    st.markdown("Analyze multiple debtors and rank by priority.")
    st.markdown("---")

    n_debtors = st.slider("Number of debtors to analyze", 5, 50, 10)

    if st.button("🚀 Run Batch Analysis", type="primary"):
        with st.spinner(f"Analyzing {n_debtors} debtors across all agents..."):
            start_time = time.time()
            batch_results = api.batch_analyze(n=n_debtors)
            elapsed = time.time() - start_time

        st.success(f"✅ Analyzed {len(batch_results)} debtors in {elapsed:.1f} seconds")

        # Summary metrics
        valid = [r for r in batch_results if 'priority_score' in r]

        if valid:
            priorities = [r['priority_score'] for r in valid]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Debtors Analyzed", len(valid))
            with col2:
                st.metric("Avg Priority", f"{np.mean(priorities):.1f}")
            with col3:
                st.metric("Highest Priority", f"{max(priorities):.1f}")
            with col4:
                st.metric("Processing Time", f"{elapsed:.1f}s")

            st.markdown("---")

            # Results table
            rows = []
            df = api.orchestrator.debtor_data

            for result in valid:
                did = result.get('debtor_id', '')
                risk = result.get('agents_results', {}).get('risk_assessment', {})
                strategy = result.get('agents_results', {}).get('strategy', {})
                compliance = result.get('agents_results', {}).get('compliance', {})
                segment = result.get('agents_results', {}).get('segmentation', {})

                mask = df['debtor_id'] == did
                if mask.any():
                    debtor = df[mask].iloc[0]
                    name = f"{debtor['first_name']} {debtor['last_name']}"
                    balance = debtor['remaining_balance']
                    dpd = debtor['days_past_due']
                else:
                    name = "Unknown"
                    balance = 0
                    dpd = 0

                rows.append({
                    'Priority': result.get('priority_score', 0),
                    'Debtor ID': did,
                    'Name': name,
                    'Balance': f"${balance:,.2f}",
                    'DPD': dpd,
                    'Risk Level': risk.get('risk_level', 'N/A'),
                    'Pay Prob': f"{risk.get('payment_probability', 0):.0%}",
                    'Strategy': strategy.get('recommended_strategy', 'N/A'),
                    'Channel': strategy.get('recommended_channel', {}).get('primary', 'N/A'),
                    'Compliant': "✅" if compliance.get('is_approved', False) else "❌",
                    'Segment': segment.get('segment_name', 'N/A'),
                })

            results_df = pd.DataFrame(rows).sort_values('Priority', ascending=False)
            st.dataframe(results_df, use_container_width=True, hide_index=True, height=500)

            # Strategy distribution
            st.markdown("---")
            st.subheader("Strategy Distribution")
            strat_counts = results_df['Strategy'].value_counts()
            st.bar_chart(strat_counts)


# ─────────────────────────────────────────────────────────
# Page: Compliance Checker
# ─────────────────────────────────────────────────────────
def page_compliance(api: DebtCollectionAPI):
    st.title("⚖️ Compliance Checker")
    st.markdown("Check proposed collection actions against FDCPA, TCPA, SCRA and internal rules.")
    st.markdown("---")

    st.subheader("Proposed Action")

    col_a, col_b = st.columns(2)
    with col_a:
        channel = st.selectbox("Contact Channel", ["phone", "email", "sms", "letter"])
        contact_time = st.time_input("Contact Time", value=None)
        time_str = contact_time.strftime("%H:%M") if contact_time else "10:00"
    with col_b:
        message = st.text_area("Message Content", height=150,
                               placeholder="Enter the message you plan to send...")

    st.subheader("Debtor Flags")

    col1, col2, col3 = st.columns(3)
    with col1:
        do_not_call = st.checkbox("Do Not Call")
        cease_desist = st.checkbox("Cease and Desist")
    with col2:
        has_attorney = st.checkbox("Represented by Attorney")
        bankruptcy = st.checkbox("Bankruptcy Filed")
    with col3:
        military = st.checkbox("Active Military")
        contacts_today = st.number_input("Contacts Today", 0, 10, 0)

    contacts_week = st.number_input("Contacts This Week", 0, 30, 0)

    if st.button("⚖️ Check Compliance", type="primary"):
        action = {
            'channel': channel,
            'time': time_str,
            'message': message or "General collection contact",
        }
        debtor = {
            'debtor_id': 'COMPLIANCE-CHECK',
            'do_not_call': do_not_call,
            'cease_and_desist': cease_desist,
            'represented_by_attorney': has_attorney,
            'bankruptcy_filed': bankruptcy,
            'active_military': military,
            'contacts_today': contacts_today,
            'contacts_this_week': contacts_week,
        }

        with st.spinner("Checking compliance..."):
            result = api.check_compliance(action, debtor)

        status = result.get('compliance_status', 'unknown')
        is_approved = result.get('is_approved', False)

        if is_approved:
            st.success(f"✅ COMPLIANT - Action is approved")
        elif status == 'warning':
            st.warning(f"⚠️ WARNING - Review required before proceeding")
        else:
            st.error(f"❌ VIOLATION - Action is BLOCKED")

        col_l, col_m, col_r = st.columns(3)
        with col_l:
            st.metric("Violations", result.get('total_violations', 0))
        with col_m:
            st.metric("Warnings", result.get('total_warnings', 0))
        with col_r:
            st.metric("Checks Passed", result.get('total_checks_passed', 0))

        if result.get('violations'):
            st.markdown("### ❌ Violations")
            for v in result['violations']:
                with st.expander(f"🔴 {v['rule']} ({v['severity']})"):
                    st.write(f"**Description:** {v['description']}")
                    st.write(f"**Action Required:** {v['action_required']}")

        if result.get('warnings'):
            st.markdown("### ⚠️ Warnings")
            for w in result['warnings']:
                st.warning(w)

        if result.get('checks_passed'):
            st.markdown("### ✅ Checks Passed")
            for c in result['checks_passed']:
                st.write(f"🟢 {c}")

        st.info(f"**Recommendation:** {result.get('recommendation', 'N/A')}")


# ─────────────────────────────────────────────────────────
# Page: Debtor Lookup
# ─────────────────────────────────────────────────────────
def page_debtor_lookup(api: DebtCollectionAPI):
    st.title("🔍 Full Debtor Analysis")
    st.markdown("Comprehensive analysis using all 7 AI agents.")
    st.markdown("---")

    df = api.orchestrator.debtor_data

    col_search, col_msg = st.columns([1, 2])

    with col_search:
        selected_id = st.selectbox("Select Debtor", df['debtor_id'].tolist()[:200])

    with col_msg:
        comm_text = st.text_input(
            "Optional: Enter a message from this debtor",
            placeholder="e.g., I want to settle this debt..."
        )

    if st.button("🧠 Run Full Analysis", type="primary"):
        with st.spinner("Running all 7 AI agents..."):
            start = time.time()
            analysis = api.analyze_debtor(
                debtor_id=selected_id,
                message=comm_text if comm_text else None
            )
            elapsed = time.time() - start

        if 'error' in analysis:
            st.error(analysis['error'])
            return

        # Header
        mask = df['debtor_id'] == selected_id
        debtor = df[mask].iloc[0] if mask.any() else None

        if debtor is not None:
            st.markdown(f"## {debtor['first_name']} {debtor['last_name']}")
            st.caption(f"ID: {selected_id} | Processed in {elapsed:.2f}s")

        # Priority and recommendation
        col_p1, col_p2 = st.columns([1, 3])
        with col_p1:
            priority = analysis.get('priority_score', 0)
            st.metric("Priority Score", f"{priority}/100")

        with col_p2:
            rec = analysis.get('final_recommendation', {})
            summary = rec.get('summary', 'No recommendation available')
            if rec.get('compliance_approved', True):
                st.success(f"📋 {summary}")
            else:
                st.warning(f"📋 {summary}")

        st.markdown("---")

        # Agent Results Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Risk", "💰 Payment", "🎮 Strategy",
            "💬 Communication", "⚖️ Compliance", "📊 Segment"
        ])

        agents = analysis.get('agents_results', {})

        with tab1:
            risk = agents.get('risk_assessment', {})
            if risk:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Risk Level", risk.get('risk_level', 'N/A').upper())
                with c2:
                    st.metric("Payment Prob", f"{risk.get('payment_probability', 0):.1%}")
                with c3:
                    st.metric("Risk Score", f"{risk.get('risk_score', 0):.4f}")
                with c4:
                    st.metric("Confidence", f"{risk.get('confidence', 0):.1%}")

        with tab2:
            payment = agents.get('payment_prediction', {})
            if payment:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Will Pay Prob", f"{payment.get('will_pay_probability', 0):.1%}")
                with c2:
                    st.metric("Predicted Amount", f"${payment.get('predicted_amount', 0):.2f}")
                with c3:
                    st.metric("Expected Value", f"${payment.get('expected_value', 0):.2f}")

        with tab3:
            strategy = agents.get('strategy', {})
            if strategy:
                st.write(f"**Recommended Strategy:** `{strategy.get('recommended_strategy', 'N/A')}`")

                channel_info = strategy.get('recommended_channel', {})
                st.write(f"**Primary Channel:** `{channel_info.get('primary', 'N/A')}`")

                timing = strategy.get('recommended_timing', {})
                if timing:
                    st.write(f"**Best Day:** {timing.get('best_day', 'N/A')}")
                    st.write(f"**Best Time:** {timing.get('best_time', 'N/A')}")
                    st.write(f"**Urgency:** {timing.get('urgency', 'N/A')}")

                recovery = strategy.get('expected_recovery', {})
                if recovery:
                    st.write(f"**Expected Recovery:** ${recovery.get('estimated_recovery_avg', 0):,.2f}")
                    st.write(f"**Recovery Probability:** {recovery.get('recovery_probability', 0):.1%}")

                ranking = strategy.get('strategy_ranking', [])
                if ranking:
                    st.markdown("**Strategy Ranking:**")
                    rank_df = pd.DataFrame(ranking)
                    st.dataframe(rank_df, use_container_width=True, hide_index=True)

        with tab4:
            comm = agents.get('communication', {})
            gen_response = agents.get('generated_response', {})
            if comm:
                st.json(comm)
            if gen_response:
                st.markdown("**Generated Response:**")
                st.text_area("Response", value=gen_response.get('response', ''), height=200, disabled=True)
            elif not comm_text:
                st.info("Enter a message above to see communication analysis.")

        with tab5:
            compliance = agents.get('compliance', {})
            if compliance:
                approved = compliance.get('is_approved', False)
                if approved:
                    st.success(f"✅ {compliance.get('status', 'compliant').upper()}")
                else:
                    st.error(f"❌ {compliance.get('status', 'violation').upper()}")

                if compliance.get('violations'):
                    for v in compliance['violations']:
                        st.error(f"**{v['rule']}**: {v['description']}")

                if compliance.get('warnings'):
                    for w in compliance['warnings']:
                        st.warning(w)

                st.info(compliance.get('recommendation', ''))

        with tab6:
            segment = agents.get('segmentation', {})
            if segment:
                st.write(f"**Segment:** {segment.get('segment_name', 'N/A')}")
                st.write(f"**Strategy:** {segment.get('recommended_strategy', 'N/A')}")
                st.write(f"**Confidence:** {segment.get('confidence', 0):.1%}")


# ─────────────────────────────────────────────────────────
# Page: System Status
# ─────────────────────────────────────────────────────────
def page_system_status(api: DebtCollectionAPI):
    st.title("📈 System Status")
    st.markdown("---")

    status = api.get_status()

    st.metric("System Status", "🟢 Online" if status.get('system_initialized') else "🔴 Offline")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Debtors Loaded", status['data_stats']['debtors_loaded'])
    with col2:
        st.metric("Communications Loaded", status['data_stats']['communications_loaded'])

    st.markdown("---")
    st.subheader("Agent Status")

    for agent_name, agent_info in status.get('agents', {}).items():
        trained = agent_info.get('is_trained', False)
        icon = "✅" if trained else "❌"
        with st.expander(f"{icon} {agent_name} v{agent_info.get('version', '?')}"):
            st.write(f"**Trained:** {trained}")
            history = agent_info.get('training_history', {})
            if history:
                st.json(history)


# ─────────────────────────────────────────────────────────
# Main App Router
# ─────────────────────────────────────────────────────────
def run():
    page = render_sidebar()
    api = initialize_system()

    if not st.session_state.get('initialized', False):
        st.info("⏳ System is initializing... Please wait.")
        st.stop()

    page_map = {
        "🏠 Dashboard": page_dashboard,
        "🎯 Risk Scoring": page_risk_scoring,
        "💬 Communication Analysis": page_communication,
        "📊 Batch Analysis": page_batch_analysis,
        "⚖️ Compliance Checker": page_compliance,
        "🔍 Debtor Lookup": page_debtor_lookup,
        "📈 System Status": page_system_status,
    }

    page_func = page_map.get(page, page_dashboard)
    page_func(api)


if __name__ == "__main__":
    run()

run()

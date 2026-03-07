"""
=============================================================================
MULTI-MODEL AI AGENTS FOR DEBT COLLECTION AGENCY
=============================================================================
A comprehensive AI-powered debt collection system featuring:

1. Debtor Risk Scoring Agent (ML-based risk assessment)
2. Communication Agent (NLP-powered conversation handling)
3. Payment Prediction Agent (Deep Learning time-series forecasting)
4. Strategy Optimization Agent (Reinforcement Learning for collection strategies)
5. Sentiment Analysis Agent (NLP for debtor communication analysis)
6. Document Processing Agent (OCR + NLP for document extraction)
7. Compliance Monitoring Agent (Rule-based + ML for regulatory compliance)
8. Orchestrator Agent (Coordinates all agents)

Author: AI Solutions Expert
Version: 2.0
=============================================================================
"""

import os
import json
import re
import logging
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Data & ML Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, accuracy_score,
    mean_absolute_error, mean_squared_error
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# NLP
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk_resources = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon',
                  'averaged_perceptron_tagger', 'punkt_tab']
for resource in nltk_resources:
    try:
        nltk.download(resource, quiet=True)
    except:
        pass

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Utilities
from collections import defaultdict
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debt_collection_ai.log')
    ]
)
logger = logging.getLogger('DebtCollectionAI')


# =============================================================================
# SECTION 1: DATA MODELS & ENUMERATIONS
# =============================================================================

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class DebtStatus(Enum):
    CURRENT = "current"
    PAST_DUE_30 = "past_due_30"
    PAST_DUE_60 = "past_due_60"
    PAST_DUE_90 = "past_due_90"
    PAST_DUE_120_PLUS = "past_due_120_plus"
    IN_COLLECTIONS = "in_collections"
    SETTLED = "settled"
    PAID = "paid"
    WRITTEN_OFF = "written_off"
    BANKRUPTCY = "bankruptcy"
    DISPUTED = "disputed"

class CommunicationChannel(Enum):
    PHONE = "phone"
    EMAIL = "email"
    SMS = "sms"
    LETTER = "letter"
    PORTAL = "portal"
    CHAT = "chat"

class CollectionStrategy(Enum):
    SOFT_REMINDER = "soft_reminder"
    STANDARD_FOLLOW_UP = "standard_follow_up"
    FIRM_NOTICE = "firm_notice"
    PAYMENT_PLAN_OFFER = "payment_plan_offer"
    SETTLEMENT_OFFER = "settlement_offer"
    ESCALATION = "escalation"
    LEGAL_WARNING = "legal_warning"
    LEGAL_ACTION = "legal_action"
    SKIP_TRACING = "skip_tracing"
    HARDSHIP_PROGRAM = "hardship_program"

class SentimentCategory(Enum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"
    HOSTILE = "hostile"
    COOPERATIVE = "cooperative"
    DISTRESSED = "distressed"

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    REVIEW_REQUIRED = "review_required"


@dataclass
class DebtorProfile:
    """Comprehensive debtor profile data model"""
    debtor_id: str
    first_name: str
    last_name: str
    email: str = ""
    phone: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""

    # Financial Information
    total_debt: float = 0.0
    original_debt: float = 0.0
    interest_accrued: float = 0.0
    fees: float = 0.0
    payments_made: float = 0.0
    remaining_balance: float = 0.0

    # Account Information
    account_number: str = ""
    creditor_name: str = ""
    debt_type: str = ""  # credit_card, medical, auto, mortgage, student_loan, personal
    account_open_date: str = ""
    last_payment_date: str = ""
    delinquency_date: str = ""
    days_past_due: int = 0
    status: str = "in_collections"

    # Credit & Scoring
    credit_score: int = 0
    income_estimate: float = 0.0
    employment_status: str = ""
    debt_to_income_ratio: float = 0.0

    # Collection History
    num_contact_attempts: int = 0
    num_successful_contacts: int = 0
    num_promises_to_pay: int = 0
    num_broken_promises: int = 0
    num_disputes: int = 0
    last_contact_date: str = ""
    preferred_contact_method: str = ""
    best_contact_time: str = ""

    # Behavioral Data
    response_rate: float = 0.0
    average_response_time_hours: float = 0.0
    sentiment_history: List[str] = field(default_factory=list)
    payment_history: List[Dict] = field(default_factory=list)
    communication_log: List[Dict] = field(default_factory=list)

    # Risk & Strategy
    risk_score: float = 0.0
    risk_level: str = "medium"
    assigned_strategy: str = ""
    priority_score: float = 0.0

    # Compliance
    do_not_call: bool = False
    cease_and_desist: bool = False
    represented_by_attorney: bool = False
    bankruptcy_filed: bool = False
    active_military: bool = False
    timezone: str = "EST"
    language_preference: str = "English"

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_feature_vector(self) -> Dict:
        """Extract ML-ready features from profile"""
        return {
            'total_debt': self.total_debt,
            'original_debt': self.original_debt,
            'payments_made': self.payments_made,
            'remaining_balance': self.remaining_balance,
            'days_past_due': self.days_past_due,
            'credit_score': self.credit_score,
            'income_estimate': self.income_estimate,
            'debt_to_income_ratio': self.debt_to_income_ratio,
            'num_contact_attempts': self.num_contact_attempts,
            'num_successful_contacts': self.num_successful_contacts,
            'num_promises_to_pay': self.num_promises_to_pay,
            'num_broken_promises': self.num_broken_promises,
            'num_disputes': self.num_disputes,
            'response_rate': self.response_rate,
            'average_response_time_hours': self.average_response_time_hours,
            'payment_ratio': self.payments_made / max(self.total_debt, 1),
            'promise_keep_rate': (
                (self.num_promises_to_pay - self.num_broken_promises) /
                max(self.num_promises_to_pay, 1)
            ),
            'contact_success_rate': (
                self.num_successful_contacts / max(self.num_contact_attempts, 1)
            ),
        }


@dataclass
class CollectionAction:
    """Represents a recommended collection action"""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    debtor_id: str = ""
    strategy: str = ""
    channel: str = ""
    priority: int = 0
    confidence: float = 0.0
    recommended_date: str = ""
    recommended_time: str = ""
    message_template: str = ""
    personalized_message: str = ""
    expected_recovery_amount: float = 0.0
    expected_recovery_probability: float = 0.0
    compliance_approved: bool = False
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentResult:
    """Standard result format for all agents"""
    agent_name: str
    success: bool
    data: Dict = field(default_factory=dict)
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# SECTION 2: DATA GENERATION & PREPROCESSING
# =============================================================================

class DebtCollectionDataGenerator:
    """Generates realistic synthetic data for training and testing"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer",
            "Michael", "Linda", "William", "Elizabeth", "David", "Barbara",
            "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
            "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
            "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra",
            "Donald", "Ashley", "Steven", "Dorothy", "Paul", "Kimberly",
            "Andrew", "Emily", "Joshua", "Donna", "Kenneth", "Michelle",
            "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
            "Timothy", "Deborah"
        ]
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
            "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez",
            "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor",
            "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
            "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis",
            "Robinson", "Walker", "Young", "Allen", "King", "Wright",
            "Scott", "Torres", "Nguyen", "Hill", "Flores"
        ]
        self.states = [
            "CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI",
            "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"
        ]
        self.debt_types = [
            "credit_card", "medical", "auto_loan", "personal_loan",
            "student_loan", "utility", "telecom", "retail"
        ]
        self.creditors = [
            "Chase Bank", "Bank of America", "Wells Fargo", "Citibank",
            "Capital One", "Discover", "American Express", "US Bank",
            "Regional Medical Center", "City Hospital", "AutoFinance Corp",
            "StudentLoan Services", "Utility Co", "TeleCom Inc"
        ]
        self.employment_statuses = [
            "employed_full_time", "employed_part_time", "self_employed",
            "unemployed", "retired", "disabled", "student"
        ]

    def generate_debtor_profiles(self, n: int = 1000) -> pd.DataFrame:
        """Generate n synthetic debtor profiles"""
        logger.info(f"Generating {n} synthetic debtor profiles...")

        profiles = []
        for i in range(n):
            # Base demographics
            first_name = np.random.choice(self.first_names)
            last_name = np.random.choice(self.last_names)
            state = np.random.choice(self.states)
            debt_type = np.random.choice(self.debt_types)
            employment = np.random.choice(
                self.employment_statuses,
                p=[0.45, 0.15, 0.10, 0.15, 0.05, 0.05, 0.05]
            )

            # Financial data with realistic correlations
            if debt_type == "medical":
                original_debt = np.random.lognormal(mean=8.0, sigma=1.2)
            elif debt_type == "credit_card":
                original_debt = np.random.lognormal(mean=8.5, sigma=0.8)
            elif debt_type in ["auto_loan", "student_loan"]:
                original_debt = np.random.lognormal(mean=9.5, sigma=0.6)
            else:
                original_debt = np.random.lognormal(mean=7.5, sigma=1.0)

            original_debt = min(original_debt, 250000)
            original_debt = round(original_debt, 2)

            # Days past due influences many other variables
            days_past_due = int(np.random.choice(
                [30, 60, 90, 120, 150, 180, 240, 300, 365, 500, 730],
                p=[0.10, 0.12, 0.15, 0.13, 0.10, 0.10, 0.08, 0.08, 0.06, 0.05, 0.03]
            ) + np.random.randint(-15, 15))
            days_past_due = max(1, days_past_due)

            # Interest and fees
            annual_rate = np.random.uniform(0.05, 0.30)
            interest_accrued = round(
                original_debt * annual_rate * (days_past_due / 365), 2
            )
            fees = round(np.random.uniform(25, 500), 2)
            total_debt = round(original_debt + interest_accrued + fees, 2)

            # Payment history
            payment_probability = max(0.05, 1.0 - (days_past_due / 1000))
            if employment in ["employed_full_time", "self_employed"]:
                payment_probability *= 1.3
            elif employment == "unemployed":
                payment_probability *= 0.5

            payment_probability = min(0.95, payment_probability)
            num_payments = np.random.binomial(12, payment_probability)
            payments_made = round(
                sum(np.random.uniform(50, total_debt * 0.1)
                    for _ in range(num_payments)), 2
            )
            payments_made = min(payments_made, total_debt * 0.8)
            remaining_balance = round(total_debt - payments_made, 2)

            # Credit score (correlated with debt status)
            base_credit = np.random.normal(680, 80)
            credit_penalty = days_past_due * 0.3 + (
                total_debt / 1000
            ) * 2
            credit_score = int(np.clip(base_credit - credit_penalty, 300, 850))

            # Income estimate
            if employment == "employed_full_time":
                income = np.random.lognormal(mean=10.8, sigma=0.5)
            elif employment == "employed_part_time":
                income = np.random.lognormal(mean=10.2, sigma=0.5)
            elif employment == "self_employed":
                income = np.random.lognormal(mean=10.9, sigma=0.7)
            elif employment == "retired":
                income = np.random.lognormal(mean=10.3, sigma=0.4)
            else:
                income = np.random.lognormal(mean=9.5, sigma=0.8)
            income = round(min(income, 500000), 2)

            dti = round(total_debt / max(income, 1) * 100, 2)

            # Contact history
            num_contact_attempts = int(np.random.poisson(
                lam=max(3, days_past_due / 30)
            ))
            num_contact_attempts = min(num_contact_attempts, 50)

            contact_success_rate = np.random.beta(2, 3)
            num_successful = int(num_contact_attempts * contact_success_rate)

            num_promises = int(num_successful * np.random.beta(2, 5))
            broken_promise_rate = np.random.beta(3, 2)
            num_broken = int(num_promises * broken_promise_rate)

            num_disputes = np.random.choice(
                [0, 1, 2, 3],
                p=[0.70, 0.15, 0.10, 0.05]
            )

            response_rate = round(
                num_successful / max(num_contact_attempts, 1), 3
            )
            avg_response_time = round(np.random.lognormal(mean=3, sigma=1), 1)

            # Compliance flags
            do_not_call = np.random.random() < 0.05
            cease_desist = np.random.random() < 0.03
            has_attorney = np.random.random() < 0.08
            bankruptcy = np.random.random() < 0.04
            military = np.random.random() < 0.02

            # Determine status
            if bankruptcy:
                status = "bankruptcy"
            elif payments_made >= total_debt * 0.95:
                status = "paid"
            elif payments_made >= total_debt * 0.5:
                status = "settled"
            elif days_past_due > 365:
                status = np.random.choice(
                    ["in_collections", "written_off"],
                    p=[0.7, 0.3]
                )
            elif days_past_due > 120:
                status = "past_due_120_plus"
            elif days_past_due > 90:
                status = "past_due_90"
            elif days_past_due > 60:
                status = "past_due_60"
            elif days_past_due > 30:
                status = "past_due_30"
            else:
                status = "current"

            # Target: Will pay within 30 days? (for ML training)
            pay_probability = (
                0.3 * (credit_score / 850) +
                0.2 * (income / 100000) +
                0.15 * response_rate +
                0.15 * (1 - min(days_past_due, 365) / 365) +
                0.1 * (payments_made / max(total_debt, 1)) +
                0.1 * (1 - num_broken / max(num_promises, 1))
            )
            pay_probability = np.clip(pay_probability + np.random.normal(0, 0.1), 0, 1)
            will_pay = 1 if np.random.random() < pay_probability else 0

            # Settlement acceptance probability
            settlement_prob = np.clip(
                0.4 + 0.2 * (days_past_due / 365) -
                0.1 * (income / 100000) +
                0.1 * response_rate + np.random.normal(0, 0.1),
                0, 1
            )

            profile = {
                'debtor_id': f"DBT-{i+1:06d}",
                'first_name': first_name,
                'last_name': last_name,
                'email': f"{first_name.lower()}.{last_name.lower()}{np.random.randint(1,99)}@email.com",
                'phone': f"+1{np.random.randint(200,999)}{np.random.randint(1000000,9999999)}",
                'state': state,
                'debt_type': debt_type,
                'creditor_name': np.random.choice(self.creditors),
                'employment_status': employment,
                'original_debt': original_debt,
                'interest_accrued': interest_accrued,
                'fees': fees,
                'total_debt': total_debt,
                'payments_made': payments_made,
                'remaining_balance': remaining_balance,
                'days_past_due': days_past_due,
                'credit_score': credit_score,
                'income_estimate': income,
                'debt_to_income_ratio': dti,
                'num_contact_attempts': num_contact_attempts,
                'num_successful_contacts': num_successful,
                'num_promises_to_pay': num_promises,
                'num_broken_promises': num_broken,
                'num_disputes': num_disputes,
                'response_rate': response_rate,
                'average_response_time_hours': avg_response_time,
                'status': status,
                'do_not_call': do_not_call,
                'cease_and_desist': cease_desist,
                'represented_by_attorney': has_attorney,
                'bankruptcy_filed': bankruptcy,
                'active_military': military,
                'will_pay_30_days': will_pay,
                'settlement_acceptance_prob': round(settlement_prob, 3),
                'pay_probability': round(pay_probability, 3),
            }
            profiles.append(profile)

        df = pd.DataFrame(profiles)
        logger.info(f"Generated {len(df)} debtor profiles")
        logger.info(f"Payment rate: {df['will_pay_30_days'].mean():.2%}")
        return df

    def generate_communication_data(self, n: int = 2000) -> pd.DataFrame:
        """Generate synthetic communication/conversation data"""
        logger.info(f"Generating {n} synthetic communication records...")

        templates = {
            'cooperative': [
                "I understand I owe this amount. Can we set up a payment plan?",
                "I want to pay but I need some time. Can I make monthly payments?",
                "I can pay {amount} per month starting next week.",
                "Thank you for calling. I've been meaning to address this debt.",
                "I recently got a new job and can start making payments now.",
                "Can you send me the details? I'll make a payment this week.",
                "I appreciate your patience. Let me see what I can do.",
                "What's the minimum I can pay to get started?",
                "I'd like to settle this. What options do I have?",
                "I can make a lump sum payment if you can reduce the amount.",
            ],
            'resistant': [
                "I don't think I owe this much. Can you verify the amount?",
                "I'm not sure this is my debt. I need more information.",
                "Times are tough right now. I can't afford to pay anything.",
                "I've already paid this. Check your records again.",
                "I'm going to need to think about it. Don't call me again this week.",
                "I lost my job and have no income right now.",
                "I have other priorities. This isn't at the top of my list.",
                "I need to talk to my spouse about this first.",
                "Can you call back next month? I might have money then.",
                "I'm dealing with medical issues and can't focus on this now.",
            ],
            'hostile': [
                "Stop calling me! I know my rights!",
                "I'm going to report you to the CFPB if you keep calling.",
                "This is harassment. I'm contacting a lawyer.",
                "I don't owe you anything. Take me to court if you want.",
                "Remove my number from your system immediately!",
                "I'm recording this call. You're violating the FDCPA.",
                "I refuse to pay. This debt is not valid.",
                "You people are scammers. Leave me alone!",
                "I'm filing a complaint with the attorney general.",
                "Don't ever contact me again or I'll sue.",
            ],
            'distressed': [
                "I want to pay but I just lost my home. I don't know what to do.",
                "I'm going through a divorce and everything is falling apart.",
                "I have a serious medical condition and the bills keep piling up.",
                "I'm barely keeping food on the table. Please understand.",
                "I feel overwhelmed. I don't know how to handle all this debt.",
                "I've been trying to get back on my feet but it's so hard.",
                "I'm a single parent working two jobs. There's nothing left.",
                "I haven't been able to sleep worrying about this debt.",
                "I want to do the right thing but I genuinely can't afford it.",
                "Please, is there any hardship program available?",
            ],
            'settlement_inquiry': [
                "What's the lowest amount you'll accept to settle this?",
                "I can pay 50% of the balance today if you settle.",
                "Will you accept a settlement? I have {amount} available.",
                "If I pay in full today, can you reduce the amount?",
                "I've been saving up. Can we negotiate a settlement?",
                "I'll pay {amount} right now to close this account.",
                "What percentage of the balance would you accept?",
                "I want to settle but I need it in writing first.",
                "Can you waive the interest and fees if I pay the principal?",
                "I'm willing to settle. Let's work something out.",
            ]
        }

        records = []
        categories = list(templates.keys())
        category_probs = [0.30, 0.25, 0.15, 0.15, 0.15]

        for i in range(n):
            category = np.random.choice(categories, p=category_probs)
            text = np.random.choice(templates[category])

            # Add some variation
            amount = np.random.randint(50, 500) * 10
            text = text.replace("{amount}", f"${amount}")

            # Add noise/variation
            if np.random.random() < 0.3:
                filler_phrases = [
                    "Um, ", "Well, ", "Look, ", "Actually, ",
                    "To be honest, ", "Listen, ", "You know, "
                ]
                text = np.random.choice(filler_phrases) + text.lower()

            if np.random.random() < 0.2:
                endings = [
                    " That's all I can say.", " Please understand.",
                    " Thank you.", " I hope we can work this out.",
                    " Let me know.", " What do you think?"
                ]
                text += np.random.choice(endings)

            # Sentiment mapping
            sentiment_map = {
                'cooperative': 'positive',
                'resistant': 'negative',
                'hostile': 'very_negative',
                'distressed': 'distressed',
                'settlement_inquiry': 'cooperative'
            }

            # Intent mapping
            intent_map = {
                'cooperative': 'willing_to_pay',
                'resistant': 'reluctant',
                'hostile': 'refuses_to_pay',
                'distressed': 'hardship',
                'settlement_inquiry': 'settlement_request'
            }

            # Outcome probability
            outcome_probs = {
                'cooperative': 0.7,
                'resistant': 0.2,
                'hostile': 0.05,
                'distressed': 0.15,
                'settlement_inquiry': 0.6
            }

            resulted_in_payment = (
                1 if np.random.random() < outcome_probs[category] else 0
            )

            record = {
                'comm_id': f"COM-{i+1:06d}",
                'debtor_id': f"DBT-{np.random.randint(1, 1001):06d}",
                'text': text,
                'category': category,
                'sentiment': sentiment_map[category],
                'intent': intent_map[category],
                'channel': np.random.choice(
                    ['phone', 'email', 'sms', 'chat'],
                    p=[0.4, 0.25, 0.2, 0.15]
                ),
                'resulted_in_payment': resulted_in_payment,
                'timestamp': (
                    datetime.now() -
                    timedelta(days=np.random.randint(0, 365))
                ).isoformat(),
                'word_count': len(text.split()),
                'has_legal_mention': int(any(
                    w in text.lower() for w in
                    ['lawyer', 'attorney', 'court', 'sue', 'legal',
                     'fdcpa', 'cfpb', 'rights', 'harassment']
                )),
                'has_payment_mention': int(any(
                    w in text.lower() for w in
                    ['pay', 'payment', 'settle', 'amount', 'money',
                     'afford', '$', 'dollar']
                )),
                'has_hardship_mention': int(any(
                    w in text.lower() for w in
                    ['lost', 'medical', 'divorce', 'job', 'sick',
                     'homeless', 'struggle', 'overwhelm', 'can\'t afford']
                )),
            }
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} communication records")
        return df

    def generate_payment_history(self, n_debtors: int = 1000,
                                  max_months: int = 24) -> pd.DataFrame:
        """Generate payment time-series data"""
        logger.info("Generating payment history data...")

        records = []
        for i in range(n_debtors):
            debtor_id = f"DBT-{i+1:06d}"
            base_amount = np.random.lognormal(mean=8, sigma=1)
            payment_consistency = np.random.beta(2, 5)

            for month in range(max_months):
                date = datetime.now() - timedelta(days=(max_months - month) * 30)

                # Simulate payment patterns
                will_pay = np.random.random() < payment_consistency

                if will_pay:
                    amount = round(
                        np.random.uniform(0.02, 0.15) * base_amount, 2
                    )
                    amount = max(25, min(amount, base_amount * 0.5))
                else:
                    amount = 0

                records.append({
                    'debtor_id': debtor_id,
                    'month': month,
                    'date': date.strftime('%Y-%m-%d'),
                    'payment_amount': amount,
                    'paid': 1 if amount > 0 else 0,
                    'remaining_balance': max(
                        0, base_amount - sum(
                            r['payment_amount']
                            for r in records
                            if r['debtor_id'] == debtor_id
                        ) - amount
                    ),
                    'days_since_last_payment': (
                        np.random.randint(1, 90) if amount > 0 else 0
                    ),
                })

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} payment history records")
        return df


class DataPreprocessor:
    """Comprehensive data cleaning and preprocessing pipeline"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_stats = {}
        self.is_fitted = False

    def clean_dataset(self, df: pd.DataFrame,
                      dataset_type: str = "debtors") -> pd.DataFrame:
        """
        Master data cleaning function

        Techniques used:
        1. Missing value detection and imputation
        2. Outlier detection and capping (IQR method)
        3. Data type validation and conversion
        4. Duplicate removal
        5. Feature engineering
        6. Encoding categorical variables
        7. Normalization/Standardization
        """
        logger.info(f"Cleaning {dataset_type} dataset with {len(df)} records...")
        df = df.copy()

        # Step 1: Remove exact duplicates
        original_len = len(df)
        df = df.drop_duplicates()
        if len(df) < original_len:
            logger.info(
                f"Removed {original_len - len(df)} duplicate records"
            )

        # Step 2: Handle missing values
        df = self._handle_missing_values(df, dataset_type)

        # Step 3: Fix data types
        df = self._fix_data_types(df, dataset_type)

        # Step 4: Handle outliers
        df = self._handle_outliers(df, dataset_type)

        # Step 5: Feature engineering
        df = self._engineer_features(df, dataset_type)

        # Step 6: Validate data integrity
        df = self._validate_data(df, dataset_type)

        logger.info(f"Cleaning complete. Final dataset: {len(df)} records, "
                    f"{len(df.columns)} columns")
        return df

    def _handle_missing_values(self, df: pd.DataFrame,
                                dataset_type: str) -> pd.DataFrame:
        """Smart missing value imputation"""
        missing_report = df.isnull().sum()
        missing_cols = missing_report[missing_report > 0]

        if len(missing_cols) > 0:
            logger.info(f"Missing values found in {len(missing_cols)} columns")

            for col in missing_cols.index:
                missing_pct = missing_cols[col] / len(df) * 100
                logger.info(f"  {col}: {missing_pct:.1f}% missing")

                if missing_pct > 50:
                    logger.warning(
                        f"  Dropping {col} (>50% missing)"
                    )
                    df = df.drop(columns=[col])
                    continue

                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    # Use median for numeric (robust to outliers)
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self.imputers[col] = ('median', median_val)
                elif df[col].dtype == 'object':
                    # Use mode for categorical
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'
                    df[col] = df[col].fillna(mode_val)
                    self.imputers[col] = ('mode', mode_val)
                elif df[col].dtype == 'bool':
                    df[col] = df[col].fillna(False)
                    self.imputers[col] = ('constant', False)

        return df

    def _fix_data_types(self, df: pd.DataFrame,
                        dataset_type: str) -> pd.DataFrame:
        """Ensure correct data types"""
        numeric_cols = [
            'total_debt', 'original_debt', 'interest_accrued', 'fees',
            'payments_made', 'remaining_balance', 'credit_score',
            'income_estimate', 'debt_to_income_ratio', 'response_rate',
            'average_response_time_hours', 'pay_probability',
            'settlement_acceptance_prob'
        ]
        int_cols = [
            'days_past_due', 'num_contact_attempts',
            'num_successful_contacts', 'num_promises_to_pay',
            'num_broken_promises', 'num_disputes', 'will_pay_30_days'
        ]
        bool_cols = [
            'do_not_call', 'cease_and_desist', 'represented_by_attorney',
            'bankruptcy_filed', 'active_military'
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col], errors='coerce'
                ).fillna(0).astype(int)

        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        return df

    def _handle_outliers(self, df: pd.DataFrame,
                         dataset_type: str) -> pd.DataFrame:
        """Cap outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_cols = [
            c for c in numeric_cols
            if c not in [
                'debtor_id', 'will_pay_30_days', 'credit_score',
                'resulted_in_payment', 'paid'
            ]
        ]

        for col in outlier_cols:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
            if outlier_count > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)

            self.feature_stats[col] = {
                'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
                'lower_bound': lower, 'upper_bound': upper
            }

        return df

    def _engineer_features(self, df: pd.DataFrame,
                            dataset_type: str) -> pd.DataFrame:
        """Create derived features"""
        if dataset_type == "debtors":
            if all(c in df.columns for c in
                   ['payments_made', 'total_debt']):
                df['payment_ratio'] = (
                    df['payments_made'] /
                    df['total_debt'].clip(lower=1)
                ).round(4)

            if all(c in df.columns for c in
                   ['num_successful_contacts', 'num_contact_attempts']):
                df['contact_success_rate'] = (
                    df['num_successful_contacts'] /
                    df['num_contact_attempts'].clip(lower=1)
                ).round(4)

            if all(c in df.columns for c in
                   ['num_promises_to_pay', 'num_broken_promises']):
                df['promise_reliability'] = (
                    1 - df['num_broken_promises'] /
                    df['num_promises_to_pay'].clip(lower=1)
                ).clip(0, 1).round(4)

            if 'days_past_due' in df.columns:
                df['dpd_bucket'] = pd.cut(
                    df['days_past_due'],
                    bins=[0, 30, 60, 90, 120, 180, 365, float('inf')],
                    labels=[
                        '0-30', '31-60', '61-90', '91-120',
                        '121-180', '181-365', '365+'
                    ]
                )

            if 'credit_score' in df.columns:
                df['credit_tier'] = pd.cut(
                    df['credit_score'],
                    bins=[0, 500, 580, 670, 740, 800, 850],
                    labels=[
                        'deep_subprime', 'subprime', 'near_prime',
                        'prime', 'prime_plus', 'super_prime'
                    ]
                )

            if all(c in df.columns for c in
                   ['remaining_balance', 'income_estimate']):
                df['affordability_index'] = (
                    df['remaining_balance'] /
                    df['income_estimate'].clip(lower=1)
                ).round(4)

            if all(c in df.columns for c in
                   ['interest_accrued', 'original_debt']):
                df['interest_burden'] = (
                    df['interest_accrued'] /
                    df['original_debt'].clip(lower=1)
                ).round(4)

        return df

    def _validate_data(self, df: pd.DataFrame,
                       dataset_type: str) -> pd.DataFrame:
        """Validate data integrity"""
        if dataset_type == "debtors":
            # Ensure non-negative financial values
            financial_cols = [
                'total_debt', 'original_debt', 'payments_made',
                'remaining_balance', 'income_estimate'
            ]
            for col in financial_cols:
                if col in df.columns:
                    df[col] = df[col].clip(lower=0)

            # Credit score bounds
            if 'credit_score' in df.columns:
                df['credit_score'] = df['credit_score'].clip(300, 850)

            # Rate bounds
            rate_cols = [
                'response_rate', 'payment_ratio',
                'contact_success_rate', 'promise_reliability'
            ]
            for col in rate_cols:
                if col in df.columns:
                    df[col] = df[col].clip(0, 1)

        return df

    def prepare_ml_features(self, df: pd.DataFrame,
                             target_col: str = 'will_pay_30_days',
                             fit: bool = True) -> Tuple[np.ndarray, np.ndarray,
                                                         List[str]]:
        """Prepare features and target for ML models"""
        # Select numeric features
        feature_cols = [
            'total_debt', 'original_debt', 'payments_made',
            'remaining_balance', 'days_past_due', 'credit_score',
            'income_estimate', 'debt_to_income_ratio',
            'num_contact_attempts', 'num_successful_contacts',
            'num_promises_to_pay', 'num_broken_promises',
            'num_disputes', 'response_rate',
            'average_response_time_hours'
        ]

        # Add engineered features if available
        engineered = [
            'payment_ratio', 'contact_success_rate',
            'promise_reliability', 'affordability_index',
            'interest_burden'
        ]
        for col in engineered:
            if col in df.columns:
                feature_cols.append(col)

        available_features = [c for c in feature_cols if c in df.columns]

        X = df[available_features].values
        y = df[target_col].values if target_col in df.columns else None

        # Encode categorical features
        categorical_cols = ['debt_type', 'employment_status', 'status']
        for cat_col in categorical_cols:
            if cat_col in df.columns:
                if fit:
                    le = LabelEncoder()
                    encoded = le.fit_transform(df[cat_col].astype(str))
                    self.encoders[cat_col] = le
                else:
                    le = self.encoders.get(cat_col)
                    if le:
                        # Handle unseen labels
                        known_classes = set(le.classes_)
                        encoded = df[cat_col].astype(str).map(
                            lambda x: x if x in known_classes else le.classes_[0]
                        )
                        encoded = le.transform(encoded)
                    else:
                        continue

                X = np.column_stack([X, encoded])
                available_features.append(cat_col)

        # Boolean features
        bool_cols = [
            'do_not_call', 'cease_and_desist',
            'represented_by_attorney', 'bankruptcy_filed',
            'active_military'
        ]
        for bool_col in bool_cols:
            if bool_col in df.columns:
                X = np.column_stack([X, df[bool_col].astype(int).values])
                available_features.append(bool_col)

        # Handle any remaining NaN
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        if fit:
            self.scalers['features'] = StandardScaler()
            X = self.scalers['features'].fit_transform(X)
            self.is_fitted = True
        elif 'features' in self.scalers:
            X = self.scalers['features'].transform(X)

        return X, y, available_features


# =============================================================================
# SECTION 3: BASE AGENT CLASS
# =============================================================================

class BaseAgent(ABC):
    """Abstract base class for all AI agents"""

    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.is_trained = False
        self.training_history = {}
        self.logger = logging.getLogger(f'Agent.{name}')

    @abstractmethod
    def train(self, data: Any, **kwargs) -> Dict:
        """Train the agent"""
        pass

    @abstractmethod
    def predict(self, input_data: Any, **kwargs) -> AgentResult:
        """Make predictions"""
        pass

    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            'name': self.name,
            'version': self.version,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
        }

    def _measure_time(self, func, *args, **kwargs):
        """Measure execution time of a function"""
        start = datetime.now()
        result = func(*args, **kwargs)
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return result, elapsed


# =============================================================================
# SECTION 4: RISK SCORING AGENT
# =============================================================================

class RiskScoringAgent(BaseAgent):
    """
    ML-based debtor risk scoring agent.
    Uses ensemble methods to predict payment likelihood and assign risk levels.
    """

    def __init__(self):
        super().__init__("RiskScoringAgent", "2.0")
        self.models = {}
        self.ensemble = None
        self.feature_importance = {}
        self.risk_thresholds = {
            'very_low': 0.8,
            'low': 0.6,
            'medium': 0.4,
            'high': 0.2,
            'very_high': 0.0
        }

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: List[str] = None, **kwargs) -> Dict:
        """Train ensemble risk scoring models"""
        self.logger.info("Training Risk Scoring Agent...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Model 1: Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf

        # Model 2: Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb

        # Model 3: Logistic Regression
        lr = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            C=1.0
        )
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr

        # Ensemble: Stacking Classifier
        estimators = [
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ]
        self.ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5,
            n_jobs=-1
        )
        self.ensemble.fit(X_train, y_train)

        # Evaluate
        results = {}
        for model_name, model in {**self.models,
                                   'ensemble': self.ensemble}.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            results[model_name] = {
                'accuracy': round(accuracy_score(y_test, y_pred), 4),
                'f1_score': round(f1_score(y_test, y_pred), 4),
                'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
            }
            self.logger.info(
                f"  {model_name}: AUC={results[model_name]['roc_auc']:.4f}, "
                f"F1={results[model_name]['f1_score']:.4f}"
            )

        # Feature importance (from Random Forest)
        if feature_names:
            importances = rf.feature_importances_
            self.feature_importance = dict(
                sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )
            )
            self.logger.info("Top 10 features:")
            for feat, imp in list(self.feature_importance.items())[:10]:
                self.logger.info(f"  {feat}: {imp:.4f}")

        # Cross-validation
        cv_scores = cross_val_score(
            self.ensemble, X, y, cv=5, scoring='roc_auc', n_jobs=-1
        )
        results['cross_validation'] = {
            'mean_auc': round(cv_scores.mean(), 4),
            'std_auc': round(cv_scores.std(), 4),
        }

        self.is_trained = True
        self.training_history = {
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'results': results
        }

        return results

    def predict(self, X: np.ndarray, **kwargs) -> AgentResult:
        """Predict risk scores for debtors"""
        if not self.is_trained:
            return AgentResult(
                agent_name=self.name,
                success=False,
                errors=["Model not trained"]
            )

        start = datetime.now()

        # Get probability predictions from ensemble
        probabilities = self.ensemble.predict_proba(X)[:, 1]
        predictions = self.ensemble.predict(X)

        # Calculate risk scores and levels
        risk_scores = []
        risk_levels = []
        for prob in probabilities:
            # Invert: high payment probability = low risk
            risk_score = round(1 - prob, 4)
            risk_scores.append(risk_score)

            if prob >= self.risk_thresholds['very_low']:
                risk_levels.append('very_low')
            elif prob >= self.risk_thresholds['low']:
                risk_levels.append('low')
            elif prob >= self.risk_thresholds['medium']:
                risk_levels.append('medium')
            elif prob >= self.risk_thresholds['high']:
                risk_levels.append('high')
            else:
                risk_levels.append('very_high')

        # Individual model predictions for confidence
        model_predictions = {}
        for name, model in self.models.items():
            model_predictions[name] = model.predict_proba(X)[:, 1]

        # Agreement score (confidence)
        agreement_scores = []
        for i in range(len(X)):
            preds = [model_predictions[m][i] for m in model_predictions]
            std = np.std(preds)
            agreement = 1 - min(std * 2, 1)
            agreement_scores.append(round(agreement, 4))

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                'payment_probabilities': probabilities.tolist(),
                'risk_scores': risk_scores,
                'risk_levels': risk_levels,
                'predictions': predictions.tolist(),
                'confidence_scores': agreement_scores,
                'model_predictions': {
                    k: v.tolist() for k, v in model_predictions.items()
                }
            },
            confidence=float(np.mean(agreement_scores)),
            processing_time_ms=elapsed
        )

    def score_single_debtor(self, features: np.ndarray) -> Dict:
        """Score a single debtor"""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        result = self.predict(features)
        if not result.success:
            return {'error': result.errors}

        return {
            'payment_probability': result.data['payment_probabilities'][0],
            'risk_score': result.data['risk_scores'][0],
            'risk_level': result.data['risk_levels'][0],
            'confidence': result.data['confidence_scores'][0],
        }


# =============================================================================
# SECTION 5: NLP COMMUNICATION AGENT
# =============================================================================

class CommunicationAgent(BaseAgent):
    """
    NLP-powered communication agent for:
    - Sentiment analysis of debtor communications
    - Intent classification
    - Response generation
    - Communication strategy recommendation
    """

    def __init__(self):
        super().__init__("CommunicationAgent", "2.0")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.intent_classifier = None
        self.tfidf_vectorizer = None
        self.intent_model = None
        self.lemmatizer = WordNetLemmatizer()
        self.response_templates = self._load_response_templates()
        self.tokenizer = None
        self.deep_sentiment_model = None
        self.max_sequence_length = 100

        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

    def _load_response_templates(self) -> Dict:
        """Load response templates for different scenarios"""
        return {
            'willing_to_pay': {
                'phone': [
                    "Thank you for your willingness to resolve this, {name}. "
                    "Let me help you set up a payment arrangement. "
                    "Based on your balance of ${balance}, we can offer you "
                    "a plan of ${monthly_amount} per month for {months} months. "
                    "Would that work for you?",

                    "I appreciate your cooperation, {name}. "
                    "We have several payment options available. "
                    "Would you prefer a lump-sum settlement or a monthly plan?",
                ],
                'email': [
                    "Dear {name},\n\n"
                    "Thank you for reaching out regarding your account. "
                    "We appreciate your willingness to resolve this matter.\n\n"
                    "Your current balance is ${balance}. We can offer the "
                    "following payment options:\n\n"
                    "1. Full payment: ${balance}\n"
                    "2. Settlement offer: ${settlement_amount} "
                    "(save ${savings})\n"
                    "3. Monthly plan: ${monthly_amount}/month "
                    "for {months} months\n\n"
                    "Please let us know which option works best for you.\n\n"
                    "Best regards,\n{agent_name}\n{company_name}",
                ],
                'sms': [
                    "{name}, thank you for your response. "
                    "To set up your payment plan, please call us at "
                    "{phone_number} or visit {portal_url}. "
                    "Ref: {account_number}",
                ],
            },
            'reluctant': {
                'phone': [
                    "I understand your concerns, {name}. "
                    "Let me explain your options. We want to work with you "
                    "to find a solution that fits your situation. "
                    "Have you considered a reduced payment plan?",

                    "{name}, I hear you, and I want to help. "
                    "We have flexible arrangements available, including "
                    "reduced monthly payments. May I go over some options?",
                ],
                'email': [
                    "Dear {name},\n\n"
                    "We understand that managing debt can be challenging. "
                    "We're here to work with you, not against you.\n\n"
                    "We'd like to discuss options that may make resolving "
                    "this account more manageable for you. "
                    "These options include:\n\n"
                    "- Extended payment plans with reduced monthly amounts\n"
                    "- Hardship programs for qualifying situations\n"
                    "- Settlement options at reduced balances\n\n"
                    "Please contact us at your earliest convenience.\n\n"
                    "Sincerely,\n{agent_name}",
                ],
            },
            'refuses_to_pay': {
                'phone': [
                    "I understand your position, {name}. "
                    "However, this is a valid debt obligation. "
                    "I'd encourage you to consider your options, "
                    "as we do want to resolve this amicably. "
                    "May I send you the account documentation?",
                ],
                'email': [
                    "Dear {name},\n\n"
                    "We've received your communication regarding account "
                    "{account_number}. We want to ensure you have all "
                    "the information needed to make an informed decision.\n\n"
                    "Please find enclosed the account documentation, "
                    "including the original creditor information and "
                    "balance details.\n\n"
                    "If you believe there is an error, you have the right "
                    "to dispute this debt within 30 days of receiving "
                    "this notice.\n\n"
                    "We remain available to discuss resolution options.\n\n"
                    "Sincerely,\n{agent_name}",
                ],
            },
            'hardship': {
                'phone': [
                    "I'm sorry to hear about your situation, {name}. "
                    "We have hardship programs available. "
                    "Let me connect you with our hardship team who can "
                    "review your case and potentially offer reduced "
                    "payments or temporary forbearance.",
                ],
                'email': [
                    "Dear {name},\n\n"
                    "We're sorry to learn about the difficulties you're "
                    "facing. We understand that life circumstances can "
                    "make financial obligations challenging.\n\n"
                    "We have a dedicated hardship program that may be "
                    "able to help. This program can offer:\n\n"
                    "- Temporary payment suspension\n"
                    "- Significantly reduced payment amounts\n"
                    "- Interest and fee waivers\n"
                    "- Extended repayment terms\n\n"
                    "To apply, please contact our hardship team at "
                    "{hardship_phone} or complete the application at "
                    "{hardship_url}.\n\n"
                    "We're here to help.\n\n"
                    "Sincerely,\n{agent_name}",
                ],
            },
            'settlement_request': {
                'phone': [
                    "Thank you for your interest in settling, {name}. "
                    "Based on your account, we can offer a settlement "
                    "of ${settlement_amount}, which saves you "
                    "${savings} off the current balance. "
                    "This offer is valid for {offer_days} days. "
                    "Would you like to proceed?",
                ],
                'email': [
                    "Dear {name},\n\n"
                    "Thank you for your interest in settling your account.\n\n"
                    "We're pleased to offer the following settlement:\n\n"
                    "Current Balance: ${balance}\n"
                    "Settlement Amount: ${settlement_amount}\n"
                    "You Save: ${savings}\n"
                    "Offer Valid Until: {offer_expiry}\n\n"
                    "To accept this offer, please make payment by the "
                    "expiration date. Payment can be made:\n"
                    "- Online: {portal_url}\n"
                    "- Phone: {phone_number}\n"
                    "- Mail: {mailing_address}\n\n"
                    "This settlement offer is contingent on receipt of "
                    "payment by the expiration date.\n\n"
                    "Best regards,\n{agent_name}",
                ],
            },
            'dispute': {
                'email': [
                    "Dear {name},\n\n"
                    "We have received your dispute regarding account "
                    "{account_number}. In accordance with the Fair Debt "
                    "Collection Practices Act (FDCPA), we are ceasing "
                    "all collection activity on this account while we "
                    "investigate your dispute.\n\n"
                    "We will provide you with verification of the debt "
                    "within 30 days. If you have any supporting "
                    "documentation, please send it to:\n\n"
                    "{dispute_address}\n\n"
                    "Thank you for bringing this to our attention.\n\n"
                    "Sincerely,\n{agent_name}\nCompliance Department",
                ],
            }
        }

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for NLP"""
        # Lowercase
        text = text.lower()

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z\s\'\.\!\?]', '', text)

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()

        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemma = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemma)

        return ' '.join(processed_tokens)

    def train(self, comm_data: pd.DataFrame, **kwargs) -> Dict:
        """Train intent classifier and deep sentiment model"""
        self.logger.info("Training Communication Agent...")

        results = {}

        # === Train Intent Classifier (TF-IDF + ML) ===
        self.logger.info("Training intent classifier...")

        texts = comm_data['text'].apply(self._preprocess_text).values
        intents = comm_data['intent'].values

        # TF-IDF vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        X_tfidf = self.tfidf_vectorizer.fit_transform(texts)

        # Encode intents
        self.intent_encoder = LabelEncoder()
        y_intent = self.intent_encoder.fit_transform(intents)

        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y_intent, test_size=0.2, random_state=42
        )

        # Train ensemble for intent classification
        self.intent_model = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=100, random_state=42
                )),
                ('lr', LogisticRegression(
                    max_iter=1000, random_state=42, C=10
                )),
            ],
            voting='soft',
            n_jobs=-1
        )
        self.intent_model.fit(X_train, y_train)

        y_pred = self.intent_model.predict(X_test)
        intent_report = classification_report(
            y_test, y_pred,
            target_names=self.intent_encoder.classes_,
            output_dict=True
        )
        results['intent_classification'] = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'report': intent_report,
        }
        self.logger.info(
            f"Intent classifier accuracy: "
            f"{results['intent_classification']['accuracy']:.4f}"
        )

        # === Train Deep Sentiment Model ===
        self.logger.info("Training deep sentiment model...")

        raw_texts = comm_data['text'].values
        sentiment_labels = comm_data['sentiment'].values

        self.sentiment_encoder = LabelEncoder()
        y_sent = self.sentiment_encoder.fit_transform(sentiment_labels)
        num_classes = len(self.sentiment_encoder.classes_)

        # Tokenize for deep learning
        self.tokenizer = Tokenizer(
            num_words=10000, oov_token='<OOV>'
        )
        self.tokenizer.fit_on_texts(raw_texts)
        sequences = self.tokenizer.texts_to_sequences(raw_texts)
        X_padded = pad_sequences(
            sequences, maxlen=self.max_sequence_length,
            padding='post', truncating='post'
        )

        X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
            X_padded, y_sent, test_size=0.2, random_state=42
        )

        # Build model
        self.deep_sentiment_model = self._build_sentiment_model(
            vocab_size=10000,
            embedding_dim=64,
            max_length=self.max_sequence_length,
            num_classes=num_classes
        )

        # Convert to categorical
        y_train_cat = keras.utils.to_categorical(y_train_dl, num_classes)
        y_test_cat = keras.utils.to_categorical(y_test_dl, num_classes)

        # Train
        history = self.deep_sentiment_model.fit(
            X_train_dl, y_train_cat,
            validation_data=(X_test_dl, y_test_cat),
            epochs=kwargs.get('epochs', 15),
            batch_size=32,
            callbacks=[
                callbacks.EarlyStopping(
                    patience=3, restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    factor=0.5, patience=2
                )
            ],
            verbose=0
        )

        # Evaluate
        dl_loss, dl_acc = self.deep_sentiment_model.evaluate(
            X_test_dl, y_test_cat, verbose=0
        )
        results['deep_sentiment'] = {
            'accuracy': round(dl_acc, 4),
            'loss': round(dl_loss, 4),
            'epochs_trained': len(history.history['loss']),
        }
        self.logger.info(
            f"Deep sentiment model accuracy: {dl_acc:.4f}"
        )

        self.is_trained = True
        self.training_history = {
            'trained_at': datetime.now().isoformat(),
            'results': results
        }

        return results

    def _build_sentiment_model(self, vocab_size: int, embedding_dim: int,
                                max_length: int,
                                num_classes: int) -> Model:
        """Build a deep learning model for sentiment analysis"""
        inputs = layers.Input(shape=(max_length,))

        # Embedding
        x = layers.Embedding(
            vocab_size, embedding_dim,
            input_length=max_length
        )(inputs)

        # Bidirectional LSTM
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.3)
        )(x)

        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)

        x = layers.Multiply()([x, attention])
        x = layers.GlobalAveragePooling1D()(x)

        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def predict(self, text: str, **kwargs) -> AgentResult:
        """Analyze text and provide comprehensive NLP analysis"""
        start = datetime.now()

        analysis = {
            'original_text': text,
            'preprocessed_text': self._preprocess_text(text),
        }

        # VADER Sentiment Analysis
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        analysis['vader_sentiment'] = vader_scores
        analysis['vader_category'] = self._categorize_vader(
            vader_scores['compound']
        )

        # Deep Learning Sentiment (if trained)
        if self.deep_sentiment_model and self.tokenizer:
            sequence = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(
                sequence, maxlen=self.max_sequence_length,
                padding='post', truncating='post'
            )
            dl_pred = self.deep_sentiment_model.predict(padded, verbose=0)
            dl_class_idx = np.argmax(dl_pred[0])
            dl_class = self.sentiment_encoder.inverse_transform([dl_class_idx])[0]
            dl_confidence = float(dl_pred[0][dl_class_idx])

            analysis['deep_sentiment'] = {
                'category': dl_class,
                'confidence': round(dl_confidence, 4),
                'all_probabilities': {
                    self.sentiment_encoder.inverse_transform([i])[0]:
                        round(float(dl_pred[0][i]), 4)
                    for i in range(len(dl_pred[0]))
                }
            }

        # Intent Classification (if trained)
        if self.intent_model and self.tfidf_vectorizer:
            processed = self._preprocess_text(text)
            tfidf_features = self.tfidf_vectorizer.transform([processed])
            intent_pred = self.intent_model.predict(tfidf_features)
            intent_proba = self.intent_model.predict_proba(tfidf_features)
            intent_class = self.intent_encoder.inverse_transform(intent_pred)[0]
            intent_confidence = float(np.max(intent_proba))

            analysis['intent'] = {
                'classification': intent_class,
                'confidence': round(intent_confidence, 4),
                'all_probabilities': {
                    self.intent_encoder.inverse_transform([i])[0]:
                        round(float(intent_proba[0][i]), 4)
                    for i in range(len(intent_proba[0]))
                }
            }

        # Keyword extraction
        analysis['keywords'] = self._extract_keywords(text)

        # Risk indicators in communication
        analysis['risk_indicators'] = self._detect_risk_indicators(text)

        # Compliance flags
        analysis['compliance_flags'] = self._detect_compliance_flags(text)

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return AgentResult(
            agent_name=self.name,
            success=True,
            data=analysis,
            confidence=analysis.get('deep_sentiment', {}).get(
                'confidence', vader_scores['compound']
            ),
            processing_time_ms=elapsed
        )

    def generate_response(self, intent: str, channel: str,
                           debtor_profile: Dict) -> str:
        """Generate a personalized response based on intent and context"""
        templates = self.response_templates.get(intent, {})
        channel_templates = templates.get(
            channel, templates.get('email', [""])
        )

        if not channel_templates:
            return "Thank you for contacting us. A representative will reach out to you shortly."

        template = np.random.choice(channel_templates)

        # Calculate settlement values
        balance = debtor_profile.get('remaining_balance', 0)
        settlement_pct = 0.6  # 60% of balance
        settlement_amount = round(balance * settlement_pct, 2)
        savings = round(balance - settlement_amount, 2)
        monthly_12 = round(balance / 12, 2)

        # Fill in template variables
        replacements = {
            '{name}': debtor_profile.get(
                'first_name',
                debtor_profile.get('last_name', 'Valued Customer')
            ),
            '{balance}': f"{balance:,.2f}",
            '{settlement_amount}': f"{settlement_amount:,.2f}",
            '{savings}': f"{savings:,.2f}",
            '{monthly_amount}': f"{monthly_12:,.2f}",
            '{months}': '12',
            '{account_number}': debtor_profile.get(
                'account_number', 'XXXX'
            ),
            '{agent_name}': 'Collection Services Team',
            '{company_name}': 'Professional Recovery Services',
            '{phone_number}': '1-800-555-0199',
            '{portal_url}': 'https://payments.example.com',
            '{mailing_address}': 'PO Box 1234, City, ST 12345',
            '{offer_days}': '30',
            '{offer_expiry}': (
                datetime.now() + timedelta(days=30)
            ).strftime('%B %d, %Y'),
            '{hardship_phone}': '1-800-555-0198',
            '{hardship_url}': 'https://hardship.example.com',
            '{dispute_address}': 'Disputes Dept, PO Box 5678, City, ST 12345',
        }

        for key, value in replacements.items():
            template = template.replace(key, str(value))

        return template

    def _categorize_vader(self, compound: float) -> str:
        """Categorize VADER compound score"""
        if compound >= 0.5:
            return 'very_positive'
        elif compound >= 0.05:
            return 'positive'
        elif compound > -0.05:
            return 'neutral'
        elif compound > -0.5:
            return 'negative'
        else:
            return 'very_negative'

    def _extract_keywords(self, text: str) -> Dict:
        """Extract relevant keywords from text"""
        keywords = {
            'payment_related': [],
            'legal_related': [],
            'hardship_related': [],
            'emotional_indicators': [],
            'action_words': [],
        }

        text_lower = text.lower()

        payment_words = [
            'pay', 'payment', 'settle', 'settlement', 'amount',
            'balance', 'money', 'afford', 'plan', 'monthly',
            'installment', 'lump', 'full'
        ]
        legal_words = [
            'lawyer', 'attorney', 'court', 'sue', 'legal', 'rights',
            'fdcpa', 'cfpb', 'harassment', 'violation', 'complaint',
            'dispute', 'cease', 'desist', 'recording'
        ]
        hardship_words = [
            'lost', 'job', 'medical', 'hospital', 'sick', 'divorce',
            'homeless', 'disabled', 'emergency', 'struggling',
            'overwhelmed', 'bankruptcy', 'unemployed'
        ]
        emotion_words = [
            'scared', 'worried', 'angry', 'frustrated', 'desperate',
            'hopeless', 'stressed', 'anxious', 'grateful', 'thankful',
            'sorry', 'ashamed', 'embarrassed'
        ]
        action_words = [
            'call', 'send', 'email', 'visit', 'contact', 'stop',
            'remove', 'delete', 'verify', 'confirm', 'agree',
            'refuse', 'dispute', 'accept'
        ]

        for word in payment_words:
            if word in text_lower:
                keywords['payment_related'].append(word)

        for word in legal_words:
            if word in text_lower:
                keywords['legal_related'].append(word)

        for word in hardship_words:
            if word in text_lower:
                keywords['hardship_related'].append(word)

        for word in emotion_words:
            if word in text_lower:
                keywords['emotional_indicators'].append(word)

        for word in action_words:
            if word in text_lower:
                keywords['action_words'].append(word)

        return keywords

    def _detect_risk_indicators(self, text: str) -> Dict:
        """Detect risk indicators in communication"""
        text_lower = text.lower()

        return {
            'legal_threat': any(
                w in text_lower for w in
                ['lawyer', 'attorney', 'court', 'sue', 'legal action']
            ),
            'regulatory_complaint': any(
                w in text_lower for w in
                ['cfpb', 'attorney general', 'complaint', 'report',
                 'ftc', 'consumer protection']
            ),
            'cease_desist_request': any(
                phrase in text_lower for phrase in
                ['stop calling', 'cease and desist', 'stop contacting',
                 'don\'t call', 'do not call', 'remove my number',
                 'never contact']
            ),
            'dispute_indication': any(
                w in text_lower for w in
                ['dispute', 'not my debt', 'don\'t owe', 'already paid',
                 'not valid', 'verify', 'validation']
            ),
            'hardship_indication': any(
                w in text_lower for w in
                ['lost job', 'medical', 'hospital', 'disabled',
                 'divorced', 'homeless', 'can\'t afford']
            ),
            'bankruptcy_mention': any(
                w in text_lower for w in
                ['bankruptcy', 'chapter 7', 'chapter 13',
                 'filing bankruptcy', 'bankrupt']
            ),
            'recording_mention': 'recording' in text_lower,
            'willingness_to_pay': any(
                phrase in text_lower for phrase in
                ['want to pay', 'willing to pay', 'can pay',
                 'set up payment', 'payment plan', 'settle']
            ),
        }

    def _detect_compliance_flags(self, text: str) -> List[str]:
        """Detect compliance-related flags in communication"""
        flags = []
        text_lower = text.lower()

        if any(w in text_lower for w in ['cease', 'desist', 'stop calling',
                                          'stop contacting', 'don\'t call']):
            flags.append('CEASE_AND_DESIST_REQUEST')

        if any(w in text_lower for w in ['dispute', 'not my debt',
                                          'verify debt', 'validation']):
            flags.append('DEBT_DISPUTE')

        if any(w in text_lower for w in ['bankruptcy', 'chapter 7',
                                          'chapter 13']):
            flags.append('BANKRUPTCY_MENTION')

        if any(w in text_lower for w in ['attorney', 'lawyer',
                                          'legal representation']):
            flags.append('ATTORNEY_REPRESENTATION')

        if any(w in text_lower for w in ['military', 'deployed',
                                          'service member', 'active duty']):
            flags.append('MILITARY_SERVICE')

        if any(w in text_lower for w in ['minor', 'underage', 'under 18']):
            flags.append('MINOR_DEBTOR')

        if any(w in text_lower for w in ['deceased', 'died', 'passed away',
                                          'death']):
            flags.append('DECEASED_DEBTOR')

        if any(w in text_lower for w in ['recording', 'record this call']):
            flags.append('CALL_RECORDING')

        return flags


# =============================================================================
# SECTION 6: PAYMENT PREDICTION AGENT (Deep Learning)
# =============================================================================

class PaymentPredictionAgent(BaseAgent):
    """
    Deep Learning agent for predicting:
    - Whether a debtor will make a payment
    - Expected payment amount
    - Optimal collection timing
    """

    def __init__(self):
        super().__init__("PaymentPredictionAgent", "2.0")
        self.payment_classifier = None
        self.amount_predictor = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def train(self, X: np.ndarray, y_class: np.ndarray,
              y_amount: np.ndarray = None,
              feature_names: List[str] = None, **kwargs) -> Dict:
        """Train payment prediction models"""
        self.logger.info("Training Payment Prediction Agent...")
        self.feature_names = feature_names or []

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42
        )

        results = {}

        # === Model 1: Payment Classification (Will pay / Won't pay) ===
        self.logger.info("Training payment classifier...")

        self.payment_classifier = self._build_classifier(
            input_dim=X.shape[1]
        )

        history = self.payment_classifier.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=kwargs.get('epochs', 50),
            batch_size=64,
            callbacks=[
                callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    factor=0.5, patience=3
                )
            ],
            verbose=0
        )

        # Evaluate classifier
        y_pred_prob = self.payment_classifier.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        classifier_results = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'f1_score': round(f1_score(y_test, y_pred), 4),
            'roc_auc': round(
                roc_auc_score(y_test, y_pred_prob.flatten()), 4
            ),
            'epochs_trained': len(history.history['loss']),
        }
        results['classifier'] = classifier_results
        self.logger.info(
            f"Classifier AUC: {classifier_results['roc_auc']:.4f}"
        )

        # === Model 2: Payment Amount Predictor ===
        if y_amount is not None:
            self.logger.info("Training amount predictor...")

            # Only train on samples where payment was made
            pay_mask = y_class == 1
            if pay_mask.sum() > 50:
                X_pay = X[pay_mask]
                y_amt = y_amount[pay_mask]

                X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
                    X_pay, y_amt, test_size=0.2, random_state=42
                )

                self.amount_predictor = self._build_regressor(
                    input_dim=X.shape[1]
                )

                self.amount_predictor.fit(
                    X_train_a, y_train_a,
                    validation_data=(X_test_a, y_test_a),
                    epochs=kwargs.get('epochs', 50),
                    batch_size=64,
                    callbacks=[
                        callbacks.EarlyStopping(
                            patience=5, restore_best_weights=True
                        ),
                    ],
                    verbose=0
                )

                y_pred_amt = self.amount_predictor.predict(
                    X_test_a, verbose=0
                ).flatten()
                results['amount_predictor'] = {
                    'mae': round(
                        mean_absolute_error(y_test_a, y_pred_amt), 2
                    ),
                    'rmse': round(
                        np.sqrt(mean_squared_error(y_test_a, y_pred_amt)), 2
                    ),
                }
                self.logger.info(
                    f"Amount predictor MAE: "
                    f"${results['amount_predictor']['mae']:.2f}"
                )

        self.is_trained = True
        self.training_history = {
            'trained_at': datetime.now().isoformat(),
            'results': results
        }

        return results

    def _build_classifier(self, input_dim: int) -> Model:
        """Build deep neural network for payment classification"""
        inputs = layers.Input(shape=(input_dim,))

        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Residual connection
        residual = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Add()([x, residual])
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        return model

    def _build_regressor(self, input_dim: int) -> Model:
        """Build deep neural network for payment amount prediction"""
        inputs = layers.Input(shape=(input_dim,))

        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='huber',
            metrics=['mae']
        )

        return model

    def predict(self, X: np.ndarray, **kwargs) -> AgentResult:
        """Predict payment likelihood and amount"""
        if not self.is_trained:
            return AgentResult(
                agent_name=self.name,
                success=False,
                errors=["Model not trained"]
            )

        start = datetime.now()

        # Payment probability
        pay_prob = self.payment_classifier.predict(X, verbose=0).flatten()
        pay_pred = (pay_prob > 0.5).astype(int)

        result_data = {
            'payment_probabilities': pay_prob.tolist(),
            'payment_predictions': pay_pred.tolist(),
        }

        # Amount prediction (for those predicted to pay)
        if self.amount_predictor is not None:
            amount_pred = self.amount_predictor.predict(
                X, verbose=0
            ).flatten()
            amount_pred = np.maximum(amount_pred, 0)  # No negative amounts
            result_data['predicted_amounts'] = amount_pred.tolist()

            # Expected value
            expected_values = (pay_prob * amount_pred).tolist()
            result_data['expected_values'] = expected_values

        # Confidence (based on distance from decision boundary)
        confidence = np.abs(pay_prob - 0.5) * 2  # 0 to 1
        result_data['confidence_scores'] = confidence.tolist()

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return AgentResult(
            agent_name=self.name,
            success=True,
            data=result_data,
            confidence=float(np.mean(confidence)),
            processing_time_ms=elapsed
        )


# =============================================================================
# SECTION 7: STRATEGY OPTIMIZATION AGENT
# =============================================================================

class StrategyOptimizationAgent(BaseAgent):
    """
    Reinforcement Learning-inspired agent for optimizing collection strategies.
    Uses Q-learning principles to select the best collection action for each
    debtor based on their profile and historical outcomes.
    """

    def __init__(self):
        super().__init__("StrategyOptimizationAgent", "2.0")
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.strategy_model = None
        self.state_encoder = None
        self.strategies = [s.value for s in CollectionStrategy]
        self.strategy_to_idx = {s: i for i, s in enumerate(self.strategies)}
        self.idx_to_strategy = {i: s for s, i in self.strategy_to_idx.items()}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def _discretize_state(self, features: Dict) -> str:
        """Convert continuous features to discrete state"""
        risk = features.get('risk_level', 'medium')
        dpd = features.get('days_past_due', 0)
        balance = features.get('remaining_balance', 0)
        response = features.get('response_rate', 0)
        promises = features.get('promise_reliability', 0)
        sentiment = features.get('last_sentiment', 'neutral')

        dpd_bucket = (
            'early' if dpd < 60 else
            'mid' if dpd < 120 else
            'late' if dpd < 365 else
            'very_late'
        )
        balance_bucket = (
            'low' if balance < 1000 else
            'medium' if balance < 5000 else
            'high' if balance < 20000 else
            'very_high'
        )
        response_bucket = (
            'low' if response < 0.3 else
            'medium' if response < 0.6 else
            'high'
        )
        promise_bucket = (
            'unreliable' if promises < 0.3 else
            'moderate' if promises < 0.7 else
            'reliable'
        )

        state = (
            f"{risk}|{dpd_bucket}|{balance_bucket}|"
            f"{response_bucket}|{promise_bucket}|{sentiment}"
        )
        return state

    def train(self, data: pd.DataFrame, **kwargs) -> Dict:
        """Train through simulated collection episodes"""
        self.logger.info("Training Strategy Optimization Agent...")

        num_episodes = kwargs.get('num_episodes', 10000)
        results = {'episodes': num_episodes, 'rewards': []}

        for episode in range(num_episodes):
            # Sample a random debtor profile
            idx = np.random.randint(len(data))
            debtor = data.iloc[idx]

            features = {
                'risk_level': debtor.get('risk_level', 'medium'),
                'days_past_due': debtor.get('days_past_due', 60),
                'remaining_balance': debtor.get('remaining_balance', 5000),
                'response_rate': debtor.get('response_rate', 0.3),
                'promise_reliability': debtor.get('promise_reliability', 0.5),
                'last_sentiment': debtor.get('sentiment', 'neutral'),
                'credit_score': debtor.get('credit_score', 600),
                'income_estimate': debtor.get('income_estimate', 40000),
            }

            state = self._discretize_state(features)

            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                action = np.random.choice(self.strategies)
            else:
                q_values = self.q_table[state]
                if q_values:
                    action = max(q_values, key=q_values.get)
                else:
                    action = np.random.choice(self.strategies)

            # Simulate reward based on action-state pair
            reward = self._simulate_reward(action, features)

            # Update Q-value
            next_features = features.copy()
            next_features['days_past_due'] += 30
            next_state = self._discretize_state(next_features)

            max_next_q = max(
                self.q_table[next_state].values()
            ) if self.q_table[next_state] else 0

            old_q = self.q_table[state][action]
            new_q = old_q + self.alpha * (
                reward + self.gamma * max_next_q - old_q
            )
            self.q_table[state][action] = new_q

            results['rewards'].append(reward)

            # Decay epsilon
            if episode % 1000 == 0 and self.epsilon > 0.01:
                self.epsilon *= 0.95

        # Train supervised model on Q-table for generalization
        self._train_strategy_model(data)

        avg_reward_last_1000 = np.mean(results['rewards'][-1000:])
        results['avg_reward_last_1000'] = round(avg_reward_last_1000, 4)
        results['unique_states'] = len(self.q_table)

        self.is_trained = True
        self.training_history = {
            'trained_at': datetime.now().isoformat(),
            'results': {
                k: v for k, v in results.items()
                if k != 'rewards'
            }
        }

        self.logger.info(
            f"Training complete. States: {len(self.q_table)}, "
            f"Avg reward (last 1000): {avg_reward_last_1000:.4f}"
        )

        return results

    def _simulate_reward(self, action: str, features: Dict) -> float:
        """Simulate reward for action-state pair"""
        base_reward = 0
        risk = features.get('risk_level', 'medium')
        dpd = features.get('days_past_due', 60)
        response_rate = features.get('response_rate', 0.3)
        balance = features.get('remaining_balance', 5000)
        credit = features.get('credit_score', 600)
        income = features.get('income_estimate', 40000)

        # Strategy effectiveness matrix
        effectiveness = {
            'soft_reminder': {
                'early': 0.7, 'mid': 0.3, 'late': 0.1, 'very_late': 0.05
            },
            'standard_follow_up': {
                'early': 0.6, 'mid': 0.5, 'late': 0.2, 'very_late': 0.1
            },
            'firm_notice': {
                'early': 0.3, 'mid': 0.5, 'late': 0.4, 'very_late': 0.2
            },
            'payment_plan_offer': {
                'early': 0.5, 'mid': 0.6, 'late': 0.5, 'very_late': 0.3
            },
            'settlement_offer': {
                'early': 0.2, 'mid': 0.4, 'late': 0.6, 'very_late': 0.5
            },
            'escalation': {
                'early': 0.1, 'mid': 0.3, 'late': 0.4, 'very_late': 0.3
            },
            'legal_warning': {
                'early': 0.1, 'mid': 0.2, 'late': 0.3, 'very_late': 0.4
            },
            'legal_action': {
                'early': 0.05, 'mid': 0.1, 'late': 0.2, 'very_late': 0.3
            },
            'skip_tracing': {
                'early': 0.1, 'mid': 0.2, 'late': 0.3, 'very_late': 0.4
            },
            'hardship_program': {
                'early': 0.3, 'mid': 0.4, 'late': 0.4, 'very_late': 0.3
            },
        }

        dpd_bucket = (
            'early' if dpd < 60 else
            'mid' if dpd < 120 else
            'late' if dpd < 365 else
            'very_late'
        )

        base_effectiveness = effectiveness.get(
            action, {'early': 0.2, 'mid': 0.2, 'late': 0.2, 'very_late': 0.2}
        ).get(dpd_bucket, 0.2)

        # Adjust for debtor characteristics
        if response_rate > 0.5:
            base_effectiveness *= 1.2
        elif response_rate < 0.2:
            base_effectiveness *= 0.7

        if credit > 650:
            base_effectiveness *= 1.1
        elif credit < 500:
            base_effectiveness *= 0.8

        # Affordability adjustment
        if income > 0 and balance / income < 0.1:
            base_effectiveness *= 1.3
        elif income > 0 and balance / income > 0.5:
            base_effectiveness *= 0.6

        # Calculate reward
        payment_prob = min(base_effectiveness, 1.0)
        made_payment = np.random.random() < payment_prob

        if made_payment:
            payment_amount = balance * np.random.uniform(0.05, 0.3)
            base_reward = payment_amount / 1000  # Normalize
        else:
            base_reward = -0.1  # Small penalty for unsuccessful contact

        # Cost penalty for expensive strategies
        cost_penalties = {
            'soft_reminder': 0.01,
            'standard_follow_up': 0.02,
            'firm_notice': 0.03,
            'payment_plan_offer': 0.02,
            'settlement_offer': 0.05,
            'escalation': 0.08,
            'legal_warning': 0.10,
            'legal_action': 0.25,
            'skip_tracing': 0.15,
            'hardship_program': 0.03,
        }
        base_reward -= cost_penalties.get(action, 0.05)

        # Add noise
        base_reward += np.random.normal(0, 0.05)

        return base_reward

    def _train_strategy_model(self, data: pd.DataFrame):
        """Train a supervised model for strategy recommendation"""
        # Create training data from Q-table
        X_train = []
        y_train = []

        feature_cols = [
            'days_past_due', 'remaining_balance', 'credit_score',
            'income_estimate', 'response_rate', 'num_contact_attempts',
            'num_promises_to_pay', 'num_broken_promises'
        ]

        available_cols = [c for c in feature_cols if c in data.columns]

        for idx in range(len(data)):
            debtor = data.iloc[idx]
            features = {
                'risk_level': debtor.get('risk_level', 'medium'),
                'days_past_due': debtor.get('days_past_due', 60),
                'remaining_balance': debtor.get('remaining_balance', 5000),
                'response_rate': debtor.get('response_rate', 0.3),
                'promise_reliability': debtor.get(
                    'promise_reliability', 0.5
                ),
                'last_sentiment': 'neutral',
            }

            state = self._discretize_state(features)
            if state in self.q_table and self.q_table[state]:
                best_action = max(
                    self.q_table[state],
                    key=self.q_table[state].get
                )
                feature_vector = [
                    debtor.get(c, 0) for c in available_cols
                ]
                X_train.append(feature_vector)
                action_idx = self.strategy_to_idx.get(
                    best_action, 0
                )
                y_train.append(action_idx)

        if len(X_train) > 50:
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Handle NaN
            X_train = np.nan_to_num(X_train, nan=0.0)

            self.strategy_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            self.strategy_model.fit(X_train, y_train)
            self.state_encoder = StandardScaler()
            self.state_encoder.fit(X_train)

    def predict(self, features: Dict, **kwargs) -> AgentResult:
        """Recommend optimal collection strategy"""
        start = datetime.now()

        state = self._discretize_state(features)

        # Get Q-values for current state
        q_values = dict(self.q_table[state]) if state in self.q_table else {}

        if q_values:
            # Sort strategies by Q-value
            sorted_strategies = sorted(
                q_values.items(), key=lambda x: x[1], reverse=True
            )

            best_strategy = sorted_strategies[0][0]
            best_q_value = sorted_strategies[0][1]

            # Calculate confidence based on Q-value separation
            if len(sorted_strategies) > 1:
                q_range = sorted_strategies[0][1] - sorted_strategies[-1][1]
                confidence = min(q_range / max(abs(best_q_value), 0.01), 1.0)
            else:
                confidence = 0.5
        else:
            # Fallback: rule-based strategy
            best_strategy, confidence = self._rule_based_strategy(features)
            sorted_strategies = [(best_strategy, 0)]

        # Determine optimal channel
        channel = self._recommend_channel(features)

        # Determine optimal timing
        timing = self._recommend_timing(features)

        # Calculate expected recovery
        expected_recovery = self._estimate_recovery(
            best_strategy, features
        )

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                'recommended_strategy': best_strategy,
                'strategy_ranking': [
                    {'strategy': s, 'q_value': round(q, 4)}
                    for s, q in sorted_strategies[:5]
                ],
                'recommended_channel': channel,
                'recommended_timing': timing,
                'expected_recovery': expected_recovery,
                'state': state,
            },
            confidence=round(abs(confidence), 4),
            processing_time_ms=elapsed
        )

    def _rule_based_strategy(self, features: Dict) -> Tuple[str, float]:
        """Fallback rule-based strategy selection"""
        dpd = features.get('days_past_due', 60)
        response_rate = features.get('response_rate', 0.3)
        risk = features.get('risk_level', 'medium')

        if dpd < 30:
            return 'soft_reminder', 0.7
        elif dpd < 60:
            if response_rate > 0.5:
                return 'payment_plan_offer', 0.6
            return 'standard_follow_up', 0.6
        elif dpd < 90:
            if risk in ['high', 'very_high']:
                return 'firm_notice', 0.5
            return 'payment_plan_offer', 0.5
        elif dpd < 180:
            if response_rate < 0.2:
                return 'skip_tracing', 0.4
            return 'settlement_offer', 0.5
        elif dpd < 365:
            return 'settlement_offer', 0.4
        else:
            return 'legal_warning', 0.3

    def _recommend_channel(self, features: Dict) -> Dict:
        """Recommend communication channel"""
        response_rate = features.get('response_rate', 0.3)
        preferred = features.get('preferred_contact', 'phone')

        channels = {
            'phone': 0.4,
            'email': 0.3,
            'sms': 0.2,
            'letter': 0.1,
        }

        # Adjust based on response rate
        if response_rate < 0.2:
            channels['sms'] += 0.1
            channels['email'] += 0.1
            channels['phone'] -= 0.1
        elif response_rate > 0.5:
            channels['phone'] += 0.1

        # Normalize
        total = sum(channels.values())
        channels = {k: round(v / total, 3) for k, v in channels.items()}

        best_channel = max(channels, key=channels.get)

        return {
            'primary': best_channel,
            'scores': channels,
        }

    def _recommend_timing(self, features: Dict) -> Dict:
        """Recommend optimal contact timing"""
        best_time = features.get('best_contact_time', 'morning')

        # Default timing recommendations
        timing = {
            'best_day': 'Tuesday',
            'best_time': '10:00 AM',
            'avoid_days': ['Sunday'],
            'timezone': features.get('timezone', 'EST'),
        }

        dpd = features.get('days_past_due', 60)
        if dpd < 30:
            timing['urgency'] = 'low'
            timing['wait_days'] = 7
        elif dpd < 90:
            timing['urgency'] = 'medium'
            timing['wait_days'] = 3
        else:
            timing['urgency'] = 'high'
            timing['wait_days'] = 1

        return timing

    def _estimate_recovery(self, strategy: str,
                            features: Dict) -> Dict:
        """Estimate expected recovery for given strategy"""
        balance = features.get('remaining_balance', 5000)

        recovery_rates = {
            'soft_reminder': (0.05, 0.15),
            'standard_follow_up': (0.05, 0.20),
            'firm_notice': (0.10, 0.25),
            'payment_plan_offer': (0.15, 0.40),
            'settlement_offer': (0.30, 0.60),
            'escalation': (0.10, 0.30),
            'legal_warning': (0.15, 0.35),
            'legal_action': (0.20, 0.50),
            'skip_tracing': (0.05, 0.15),
            'hardship_program': (0.10, 0.25),
        }

        low, high = recovery_rates.get(strategy, (0.05, 0.20))

        # Adjust based on response rate
        response_rate = features.get('response_rate', 0.3)
        adjustment = 1 + (response_rate - 0.3) * 0.5

        estimated_low = round(balance * low * adjustment, 2)
        estimated_high = round(balance * high * adjustment, 2)
        estimated_avg = round((estimated_low + estimated_high) / 2, 2)

        return {
            'estimated_recovery_low': estimated_low,
            'estimated_recovery_high': estimated_high,
            'estimated_recovery_avg': estimated_avg,
            'recovery_probability': round(
                (low + high) / 2 * adjustment, 4
            ),
        }


# =============================================================================
# SECTION 8: COMPLIANCE MONITORING AGENT
# =============================================================================

class ComplianceMonitoringAgent(BaseAgent):
    """
    Monitors all collection activities for regulatory compliance.
    Covers FDCPA, TCPA, state-specific regulations, and internal policies.
    """

    def __init__(self):
        super().__init__("ComplianceMonitoringAgent", "2.0")
        self.rules = self._load_compliance_rules()
        self.violation_log = []
        self.is_trained = True  # Rule-based, no training needed

    def _load_compliance_rules(self) -> Dict:
        """Load compliance rules"""
        return {
            'fdcpa': {
                'contact_hours': {
                    'earliest': 8,
                    'latest': 21,
                    'description': (
                        'FDCPA prohibits calls before 8am or after 9pm '
                        'in debtor\'s time zone'
                    ),
                },
                'contact_frequency': {
                    'max_calls_per_day': 1,
                    'max_calls_per_week': 7,
                    'max_calls_per_month': 15,
                    'description': (
                        'Regulation F limits call frequency'
                    ),
                },
                'cease_and_desist': {
                    'description': (
                        'Must cease contact upon written request '
                        '(except to notify of specific actions)'
                    ),
                },
                'validation_notice': {
                    'days_to_send': 5,
                    'dispute_period_days': 30,
                    'description': (
                        'Must send written validation notice '
                        'within 5 days of initial communication'
                    ),
                },
                'third_party_disclosure': {
                    'description': (
                        'Cannot disclose debt to third parties '
                        '(except spouse, parent of minor, '
                        'guardian, attorney, or co-signer)'
                    ),
                },
                'false_representation': {
                    'description': (
                        'Cannot use false, deceptive, or '
                        'misleading representations'
                    ),
                },
                'harassment': {
                    'description': (
                        'Cannot use harassment, oppression, '
                        'or abuse in collection attempts'
                    ),
                },
            },
            'tcpa': {
                'prior_consent': {
                    'description': (
                        'Must have prior express consent for '
                        'autodialed or prerecorded calls to cell phones'
                    ),
                },
                'do_not_call': {
                    'description': (
                        'Must honor Do Not Call registry and requests'
                    ),
                },
            },
            'scra': {
                'active_military': {
                    'max_interest_rate': 6,
                    'description': (
                        'Servicemembers Civil Relief Act limits '
                        'interest to 6% for pre-service debts'
                    ),
                },
            },
            'state_specific': {
                'mini_miranda': {
                    'description': (
                        'Many states require mini-Miranda warning '
                        'in all communications'
                    ),
                },
                'licensing': {
                    'description': (
                        'Must be licensed in debtor\'s state'
                    ),
                },
            },
        }

    def train(self, data: Any = None, **kwargs) -> Dict:
        """No training needed for rule-based agent"""
        return {'status': 'Rule-based agent, no training required'}

    def predict(self, action: Dict, debtor: Dict,
                **kwargs) -> AgentResult:
        """Check proposed action for compliance violations"""
        return self.check_compliance(action, debtor)

    def check_compliance(self, proposed_action: Dict,
                         debtor_profile: Dict) -> AgentResult:
        """
        Comprehensive compliance check for a proposed collection action
        """
        start = datetime.now()
        violations = []
        warnings = []
        checks_passed = []

        # 1. Check cease and desist
        if debtor_profile.get('cease_and_desist', False):
            if proposed_action.get('channel') in ['phone', 'sms', 'email']:
                violations.append({
                    'rule': 'FDCPA - Cease and Desist',
                    'severity': 'CRITICAL',
                    'description': (
                        'Debtor has requested cease and desist. '
                        'Only specific notifications allowed.'
                    ),
                    'action_required': (
                        'Cancel proposed contact. Only send legal '
                        'notices via mail.'
                    ),
                })
            else:
                checks_passed.append('Cease and desist - mail allowed')

        # 2. Check contact hours
        proposed_time = proposed_action.get('time', '10:00')
        if proposed_time:
            try:
                hour = int(proposed_time.split(':')[0])
                if hour < 8 or hour >= 21:
                    violations.append({
                        'rule': 'FDCPA - Contact Hours',
                        'severity': 'HIGH',
                        'description': (
                            f'Proposed contact at {proposed_time} violates '
                            f'FDCPA contact hour restrictions (8am-9pm)'
                        ),
                        'action_required': (
                            'Reschedule contact to between 8:00 AM '
                            'and 9:00 PM in debtor\'s time zone'
                        ),
                    })
                else:
                    checks_passed.append(
                        f'Contact hours ({proposed_time}) compliant'
                    )
            except:
                warnings.append('Could not validate contact time format')

        # 3. Check contact frequency
        contact_count_today = debtor_profile.get(
            'contacts_today', 0
        )
        contact_count_week = debtor_profile.get(
            'contacts_this_week', 0
        )
        contact_count_month = debtor_profile.get(
            'contacts_this_month', 0
        )

        freq_rules = self.rules['fdcpa']['contact_frequency']
        if contact_count_today >= freq_rules['max_calls_per_day']:
            violations.append({
                'rule': 'Regulation F - Daily Contact Limit',
                'severity': 'HIGH',
                'description': (
                    f'Already contacted {contact_count_today} times today. '
                    f'Max: {freq_rules["max_calls_per_day"]}'
                ),
                'action_required': 'Wait until next day to contact',
            })
        else:
            checks_passed.append('Daily contact limit compliant')

        if contact_count_week >= freq_rules['max_calls_per_week']:
            violations.append({
                'rule': 'Regulation F - Weekly Contact Limit',
                'severity': 'HIGH',
                'description': (
                    f'Already contacted {contact_count_week} times this week. '
                    f'Max: {freq_rules["max_calls_per_week"]}'
                ),
                'action_required': 'Wait until next week to contact',
            })
        else:
            checks_passed.append('Weekly contact limit compliant')

        # 4. Check Do Not Call
        if debtor_profile.get('do_not_call', False):
            if proposed_action.get('channel') == 'phone':
                violations.append({
                    'rule': 'TCPA - Do Not Call',
                    'severity': 'HIGH',
                    'description': (
                        'Debtor is on Do Not Call list'
                    ),
                    'action_required': (
                        'Use alternative contact method '
                        '(email, mail, portal)'
                    ),
                })
            else:
                checks_passed.append(
                    'Do Not Call - alternative channel used'
                )

        # 5. Check attorney representation
        if debtor_profile.get('represented_by_attorney', False):
            if proposed_action.get('contact_debtor_directly', True):
                violations.append({
                    'rule': 'FDCPA - Attorney Representation',
                    'severity': 'CRITICAL',
                    'description': (
                        'Debtor is represented by attorney. '
                        'Must contact attorney instead.'
                    ),
                    'action_required': (
                        'Redirect all communications to '
                        'debtor\'s attorney'
                    ),
                })

        # 6. Check bankruptcy
        if debtor_profile.get('bankruptcy_filed', False):
            violations.append({
                'rule': 'Automatic Stay - Bankruptcy',
                'severity': 'CRITICAL',
                'description': (
                    'Debtor has filed bankruptcy. '
                    'Automatic stay in effect.'
                ),
                'action_required': (
                    'Cease all collection activity immediately. '
                    'File proof of claim with bankruptcy court.'
                ),
            })

        # 7. Check military status (SCRA)
        if debtor_profile.get('active_military', False):
            warnings.append(
                'Active military member - SCRA protections apply. '
                'Verify interest rate cap and other protections.'
            )
            checks_passed.append('SCRA flag noted')

        # 8. Check content compliance
        message = proposed_action.get('message', '')
        if message:
            content_issues = self._check_message_compliance(message)
            violations.extend(content_issues.get('violations', []))
            warnings.extend(content_issues.get('warnings', []))

        # Determine overall status
        if any(v['severity'] == 'CRITICAL' for v in violations):
            status = ComplianceStatus.VIOLATION.value
        elif violations:
            status = ComplianceStatus.WARNING.value
        elif warnings:
            status = ComplianceStatus.REVIEW_REQUIRED.value
        else:
            status = ComplianceStatus.COMPLIANT.value

        elapsed = (datetime.now() - start).total_seconds() * 1000

        result = AgentResult(
            agent_name=self.name,
            success=True,
            data={
                'compliance_status': status,
                'violations': violations,
                'warnings': warnings,
                'checks_passed': checks_passed,
                'is_approved': len(violations) == 0,
                'total_violations': len(violations),
                'total_warnings': len(warnings),
                'total_checks_passed': len(checks_passed),
                'recommendation': self._get_recommendation(
                    violations, warnings
                ),
            },
            confidence=1.0 if len(violations) == 0 else 0.0,
            processing_time_ms=elapsed
        )

        # Log violations
        if violations:
            for v in violations:
                self.violation_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'debtor_id': debtor_profile.get('debtor_id', ''),
                    'violation': v,
                })

        return result

    def _check_message_compliance(self, message: str) -> Dict:
        """Check message content for compliance issues"""
        violations = []
        warnings = []

        message_lower = message.lower()

        # Check for prohibited language
        threatening_phrases = [
            'arrest you', 'go to jail', 'criminal charges',
            'garnish your wages without court', 'seize your property',
            'tell your employer', 'tell your family',
            'ruin your life', 'destroy your credit permanently'
        ]

        for phrase in threatening_phrases:
            if phrase in message_lower:
                violations.append({
                    'rule': 'FDCPA - False Threats',
                    'severity': 'HIGH',
                    'description': (
                        f'Message contains prohibited threatening '
                        f'language: "{phrase}"'
                    ),
                    'action_required': 'Remove prohibited language',
                })

        # Check for mini-Miranda warning (simplified check)
        if 'collect' not in message_lower and 'debt' not in message_lower:
            warnings.append(
                'Message may be missing mini-Miranda disclosure. '
                'Ensure "this is an attempt to collect a debt" is included.'
            )

        # Check for false urgency
        false_urgency = [
            'act now or else', 'last chance before legal action',
            'final warning', 'immediate legal proceedings'
        ]
        for phrase in false_urgency:
            if phrase in message_lower:
                warnings.append(
                    f'Message may contain misleading urgency: "{phrase}"'
                )

        return {'violations': violations, 'warnings': warnings}

    def _get_recommendation(self, violations: List,
                             warnings: List) -> str:
        """Generate compliance recommendation"""
        if not violations and not warnings:
            return "Action is compliant. Proceed as planned."
        elif not violations:
            return (
                f"Action has {len(warnings)} warning(s). "
                "Review warnings before proceeding."
            )
        else:
            return (
                f"Action has {len(violations)} violation(s). "
                "DO NOT proceed. Address violations first."
            )


# =============================================================================
# SECTION 9: DEBTOR SEGMENTATION AGENT
# =============================================================================

class DebtorSegmentationAgent(BaseAgent):
    """
    Uses clustering to segment debtors into actionable groups
    for targeted collection strategies.
    """

    def __init__(self, n_clusters: int = 6):
        super().__init__("DebtorSegmentationAgent", "1.0")
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.segment_profiles = {}
        self.feature_cols = []

    def train(self, df: pd.DataFrame, **kwargs) -> Dict:
        """Train segmentation model"""
        self.logger.info("Training Debtor Segmentation Agent...")

        self.feature_cols = [
            'total_debt', 'remaining_balance', 'days_past_due',
            'credit_score', 'income_estimate', 'response_rate',
            'payment_ratio', 'contact_success_rate',
            'promise_reliability'
        ]

        available = [c for c in self.feature_cols if c in df.columns]
        self.feature_cols = available

        X = df[available].values
        X = np.nan_to_num(X, nan=0.0)

        X_scaled = self.scaler.fit_transform(X)

        # Find optimal clusters using elbow method
        if kwargs.get('auto_clusters', False):
            inertias = []
            K_range = range(2, 11)
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                inertias.append(km.inertia_)

            # Simple elbow detection
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            self.n_clusters = int(np.argmax(diffs2) + 3)
            self.logger.info(
                f"Optimal clusters (elbow): {self.n_clusters}"
            )

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        clusters = self.kmeans.fit_predict(X_scaled)

        # Analyze segments
        df_analysis = df[available].copy()
        df_analysis['cluster'] = clusters

        segment_names = [
            'High-Value Cooperative',
            'Medium-Risk Payment Plan',
            'Low-Response Skip Trace',
            'Settlement Candidates',
            'Hardship Cases',
            'Legal Action Queue'
        ]

        for i in range(self.n_clusters):
            cluster_data = df_analysis[df_analysis['cluster'] == i]
            name = segment_names[i] if i < len(segment_names) else f'Segment_{i}'

            self.segment_profiles[i] = {
                'name': name,
                'size': len(cluster_data),
                'percentage': round(
                    len(cluster_data) / len(df) * 100, 1
                ),
                'profile': {
                    col: {
                        'mean': round(cluster_data[col].mean(), 2),
                        'median': round(cluster_data[col].median(), 2),
                        'std': round(cluster_data[col].std(), 2),
                    }
                    for col in available
                },
                'recommended_strategy': self._recommend_segment_strategy(
                    cluster_data, available
                ),
            }

        self.is_trained = True
        self.training_history = {
            'trained_at': datetime.now().isoformat(),
            'n_clusters': self.n_clusters,
            'segment_profiles': {
                k: {kk: vv for kk, vv in v.items() if kk != 'profile'}
                for k, v in self.segment_profiles.items()
            },
        }

        return self.training_history

    def _recommend_segment_strategy(self, cluster_data: pd.DataFrame,
                                     features: List[str]) -> str:
        """Recommend strategy for a segment"""
        avg_dpd = cluster_data.get('days_past_due', pd.Series([60])).mean()
        avg_response = cluster_data.get('response_rate', pd.Series([0.3])).mean()
        avg_balance = cluster_data.get('remaining_balance', pd.Series([5000])).mean()
        avg_payment = cluster_data.get('payment_ratio', pd.Series([0.2])).mean()

        if avg_response > 0.5 and avg_payment > 0.3:
            return 'payment_plan_offer'
        elif avg_dpd > 180 and avg_response < 0.2:
            return 'skip_tracing'
        elif avg_dpd > 120 and avg_balance > 10000:
            return 'settlement_offer'
        elif avg_response < 0.15:
            return 'escalation'
        elif avg_dpd < 60:
            return 'soft_reminder'
        else:
            return 'standard_follow_up'

    def predict(self, X: np.ndarray, **kwargs) -> AgentResult:
        """Assign debtors to segments"""
        if not self.is_trained:
            return AgentResult(
                agent_name=self.name,
                success=False,
                errors=["Model not trained"]
            )

        start = datetime.now()
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)
        clusters = self.kmeans.predict(X_scaled)
        distances = self.kmeans.transform(X_scaled)

        assignments = []
        for i, cluster in enumerate(clusters):
            profile = self.segment_profiles.get(cluster, {})
            assignments.append({
                'segment_id': int(cluster),
                'segment_name': profile.get('name', f'Segment_{cluster}'),
                'recommended_strategy': profile.get(
                    'recommended_strategy', 'standard_follow_up'
                ),
                'distance_to_center': round(
                    float(distances[i][cluster]), 4
                ),
                'confidence': round(
                    1 / (1 + float(distances[i][cluster])), 4
                ),
            })

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                'assignments': assignments,
                'segment_profiles': self.segment_profiles,
            },
            confidence=float(np.mean([a['confidence'] for a in assignments])),
            processing_time_ms=elapsed
        )


# =============================================================================
# SECTION 10: MASTER ORCHESTRATOR AGENT
# =============================================================================

class OrchestratorAgent:
    """
    Master agent that coordinates all sub-agents to provide
    comprehensive debt collection intelligence.
    """

    def __init__(self):
        self.logger = logging.getLogger('Orchestrator')
        self.logger.info("Initializing Orchestrator Agent...")

        # Initialize all agents
        self.risk_agent = RiskScoringAgent()
        self.communication_agent = CommunicationAgent()
        self.payment_agent = PaymentPredictionAgent()
        self.strategy_agent = StrategyOptimizationAgent()
        self.compliance_agent = ComplianceMonitoringAgent()
        self.segmentation_agent = DebtorSegmentationAgent()

        # Data components
        self.data_generator = DebtCollectionDataGenerator()
        self.preprocessor = DataPreprocessor()

        # State
        self.is_initialized = False
        self.agents_status = {}

    def initialize_and_train(self, n_debtors: int = 1000,
                              n_communications: int = 2000) -> Dict:
        """
        Complete system initialization:
        1. Generate training data
        2. Preprocess data
        3. Train all agents
        """
        self.logger.info("=" * 60)
        self.logger.info("INITIALIZING DEBT COLLECTION AI SYSTEM")
        self.logger.info("=" * 60)

        results = {}
        start_time = datetime.now()

        # Step 1: Generate Data
        self.logger.info("\n📊 Step 1: Generating Training Data...")
        debtor_df = self.data_generator.generate_debtor_profiles(n_debtors)
        comm_df = self.data_generator.generate_communication_data(
            n_communications
        )
        results['data_generation'] = {
            'debtors': len(debtor_df),
            'communications': len(comm_df),
        }

        # Step 2: Preprocess Data
        self.logger.info("\n🔧 Step 2: Preprocessing Data...")
        debtor_df_clean = self.preprocessor.clean_dataset(
            debtor_df, "debtors"
        )
        X, y, feature_names = self.preprocessor.prepare_ml_features(
            debtor_df_clean, 'will_pay_30_days'
        )
        results['preprocessing'] = {
            'features': len(feature_names),
            'samples': len(X),
            'feature_names': feature_names,
        }

        # Step 3: Train Risk Scoring Agent
        self.logger.info("\n🎯 Step 3: Training Risk Scoring Agent...")
        results['risk_agent'] = self.risk_agent.train(
            X, y, feature_names
        )

        # Step 4: Train Communication Agent
        self.logger.info("\n💬 Step 4: Training Communication Agent...")
        results['communication_agent'] = self.communication_agent.train(
            comm_df, epochs=10
        )

        # Step 5: Train Payment Prediction Agent
        self.logger.info("\n💰 Step 5: Training Payment Prediction Agent...")
        y_amounts = debtor_df_clean['payments_made'].values
        # Scale amounts for the model
        y_amounts_scaled = y_amounts / max(y_amounts.max(), 1)
        results['payment_agent'] = self.payment_agent.train(
            X, y, y_amounts_scaled, feature_names, epochs=20
        )

        # Step 6: Train Strategy Optimization Agent
        self.logger.info("\n🎮 Step 6: Training Strategy Agent...")
        results['strategy_agent'] = self.strategy_agent.train(
            debtor_df_clean, num_episodes=5000
        )

        # Step 7: Train Segmentation Agent
        self.logger.info("\n📊 Step 7: Training Segmentation Agent...")
        results['segmentation_agent'] = self.segmentation_agent.train(
            debtor_df_clean, auto_clusters=True
        )

        # Store clean data for reference
        self.debtor_data = debtor_df_clean
        self.comm_data = comm_df
        self.X = X
        self.y = y
        self.feature_names = feature_names

        elapsed = (datetime.now() - start_time).total_seconds()
        results['total_training_time_seconds'] = round(elapsed, 2)

        self.is_initialized = True
        self._update_agents_status()

        self.logger.info("\n" + "=" * 60)
        self.logger.info(
            f"✅ SYSTEM INITIALIZED IN {elapsed:.1f} SECONDS"
        )
        self.logger.info("=" * 60)

        return results

    def analyze_debtor(self, debtor_id: str = None,
                       debtor_profile: Dict = None,
                       communication_text: str = None) -> Dict:
        """
        Comprehensive debtor analysis using all agents.
        This is the main entry point for analyzing a debtor.
        """
        if not self.is_initialized:
            return {'error': 'System not initialized. Call initialize_and_train() first.'}

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"ANALYZING DEBTOR: {debtor_id or 'Custom Profile'}")
        self.logger.info(f"{'='*50}")

        analysis = {
            'debtor_id': debtor_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'agents_results': {},
        }

        # Get debtor data
        if debtor_id and hasattr(self, 'debtor_data'):
            mask = self.debtor_data['debtor_id'] == debtor_id
            if mask.any():
                idx = mask.values.argmax()
                debtor_row = self.debtor_data.iloc[idx]
                debtor_profile = debtor_row.to_dict()
                feature_vector = self.X[idx:idx+1]
            else:
                return {'error': f'Debtor {debtor_id} not found'}
        elif debtor_profile:
            # Create feature vector from profile
            temp_df = pd.DataFrame([debtor_profile])
            temp_df = self.preprocessor.clean_dataset(temp_df, "debtors")
            feature_vector, _, _ = self.preprocessor.prepare_ml_features(
                temp_df, fit=False
            )
        else:
            return {'error': 'Provide either debtor_id or debtor_profile'}

        # === Agent 1: Risk Scoring ===
        self.logger.info("  🎯 Running Risk Assessment...")
        risk_result = self.risk_agent.predict(feature_vector)
        if risk_result.success:
            analysis['agents_results']['risk_assessment'] = {
                'risk_score': risk_result.data['risk_scores'][0],
                'risk_level': risk_result.data['risk_levels'][0],
                'payment_probability': risk_result.data[
                    'payment_probabilities'
                ][0],
                'confidence': risk_result.data['confidence_scores'][0],
                'processing_time_ms': risk_result.processing_time_ms,
            }

        # === Agent 2: Payment Prediction ===
        self.logger.info("  💰 Running Payment Prediction...")
        payment_result = self.payment_agent.predict(feature_vector)
        if payment_result.success:
            analysis['agents_results']['payment_prediction'] = {
                'will_pay_probability': payment_result.data[
                    'payment_probabilities'
                ][0],
                'predicted_amount': payment_result.data.get(
                    'predicted_amounts', [0]
                )[0],
                'expected_value': payment_result.data.get(
                    'expected_values', [0]
                )[0],
                'confidence': payment_result.data['confidence_scores'][0],
                'processing_time_ms': payment_result.processing_time_ms,
            }

        # === Agent 3: Strategy Optimization ===
        self.logger.info("  🎮 Running Strategy Optimization...")
        strategy_features = {
            'risk_level': analysis['agents_results'].get(
                'risk_assessment', {}
            ).get('risk_level', 'medium'),
            'days_past_due': debtor_profile.get('days_past_due', 60),
            'remaining_balance': debtor_profile.get(
                'remaining_balance', 5000
            ),
            'response_rate': debtor_profile.get('response_rate', 0.3),
            'promise_reliability': debtor_profile.get(
                'promise_reliability', 0.5
            ),
            'last_sentiment': 'neutral',
            'credit_score': debtor_profile.get('credit_score', 600),
            'income_estimate': debtor_profile.get('income_estimate', 40000),
            'preferred_contact': debtor_profile.get(
                'preferred_contact_method', 'phone'
            ),
            'timezone': debtor_profile.get('timezone', 'EST'),
            'best_contact_time': debtor_profile.get(
                'best_contact_time', 'morning'
            ),
        }
        strategy_result = self.strategy_agent.predict(strategy_features)
        if strategy_result.success:
            analysis['agents_results']['strategy'] = strategy_result.data
            analysis['agents_results']['strategy'][
                'processing_time_ms'
            ] = strategy_result.processing_time_ms

        # === Agent 4: Communication Analysis ===
        if communication_text:
            self.logger.info("  💬 Running Communication Analysis...")
            comm_result = self.communication_agent.predict(
                communication_text
            )
            if comm_result.success:
                analysis['agents_results']['communication'] = comm_result.data
                analysis['agents_results']['communication'][
                    'processing_time_ms'
                ] = comm_result.processing_time_ms

                # Generate response
                intent = comm_result.data.get('intent', {}).get(
                    'classification', 'reluctant'
                )
                channel = strategy_result.data.get(
                    'recommended_channel', {}
                ).get('primary', 'email') if strategy_result.success else 'email'

                response = self.communication_agent.generate_response(
                    intent=intent,
                    channel=channel,
                    debtor_profile=debtor_profile
                )
                analysis['agents_results']['generated_response'] = {
                    'intent': intent,
                    'channel': channel,
                    'response': response,
                }

        # === Agent 5: Compliance Check ===
        self.logger.info("  ⚖️ Running Compliance Check...")
        proposed_action = {
            'channel': strategy_result.data.get(
                'recommended_channel', {}
            ).get('primary', 'email') if strategy_result.success else 'email',
            'time': '10:00',
            'message': analysis['agents_results'].get(
                'generated_response', {}
            ).get('response', ''),
        }
        compliance_result = self.compliance_agent.check_compliance(
            proposed_action, debtor_profile
        )
        if compliance_result.success:
            analysis['agents_results']['compliance'] = {
                'status': compliance_result.data['compliance_status'],
                'is_approved': compliance_result.data['is_approved'],
                'violations': compliance_result.data['violations'],
                'warnings': compliance_result.data['warnings'],
                'recommendation': compliance_result.data['recommendation'],
                'processing_time_ms': compliance_result.processing_time_ms,
            }

        # === Agent 6: Segmentation ===
        self.logger.info("  📊 Running Segmentation...")
        seg_features_cols = self.segmentation_agent.feature_cols
        seg_values = [
            debtor_profile.get(c, 0) for c in seg_features_cols
        ]
        seg_input = np.array([seg_values])
        seg_result = self.segmentation_agent.predict(seg_input)
        if seg_result.success and seg_result.data.get('assignments'):
            analysis['agents_results']['segmentation'] = (
                seg_result.data['assignments'][0]
            )

        # === Generate Final Recommendation ===
        analysis['final_recommendation'] = self._generate_recommendation(
            analysis
        )

        # === Priority Score ===
        analysis['priority_score'] = self._calculate_priority(analysis)

        self.logger.info(f"\n  ✅ Analysis Complete")
        self.logger.info(
            f"  Risk Level: "
            f"{analysis['agents_results'].get('risk_assessment', {}).get('risk_level', 'N/A')}"
        )
        self.logger.info(
            f"  Strategy: "
            f"{analysis['agents_results'].get('strategy', {}).get('recommended_strategy', 'N/A')}"
        )
        self.logger.info(
            f"  Compliance: "
            f"{analysis['agents_results'].get('compliance', {}).get('status', 'N/A')}"
        )
        self.logger.info(
            f"  Priority: {analysis['priority_score']}")

        return analysis

    def batch_analyze(self, debtor_ids: List[str] = None,
                      n_sample: int = 20) -> List[Dict]:
        """Analyze multiple debtors and return sorted by priority"""
        if not self.is_initialized:
            return [{'error': 'System not initialized'}]

        if debtor_ids is None:
            debtor_ids = self.debtor_data['debtor_id'].sample(
                min(n_sample, len(self.debtor_data))
            ).tolist()

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BATCH ANALYSIS: {len(debtor_ids)} debtors")
        self.logger.info(f"{'='*60}")

        analyses = []
        for did in debtor_ids:
            try:
                result = self.analyze_debtor(debtor_id=did)
                analyses.append(result)
            except Exception as e:
                self.logger.error(f"Error analyzing {did}: {str(e)}")
                analyses.append({
                    'debtor_id': did,
                    'error': str(e)
                })

        # Sort by priority
        valid = [a for a in analyses if 'priority_score' in a]
        valid.sort(key=lambda x: x['priority_score'], reverse=True)

        # Summary
        self.logger.info(f"\n📋 Batch Analysis Summary:")
        self.logger.info(f"  Total analyzed: {len(valid)}")

        if valid:
            priorities = [a['priority_score'] for a in valid]
            self.logger.info(
                f"  Avg priority: {np.mean(priorities):.1f}"
            )
            self.logger.info(
                f"  Highest priority: {valid[0]['debtor_id']} "
                f"(score: {valid[0]['priority_score']:.1f})"
            )

            # Strategy distribution
            strategies = [
                a.get('agents_results', {}).get(
                    'strategy', {}
                ).get('recommended_strategy', 'unknown')
                for a in valid
            ]
            strategy_counts = pd.Series(strategies).value_counts()
            self.logger.info(f"  Strategy distribution:")
            for strat, count in strategy_counts.items():
                self.logger.info(f"    {strat}: {count}")

        return valid

    def _generate_recommendation(self, analysis: Dict) -> Dict:
        """Generate final actionable recommendation"""
        risk = analysis['agents_results'].get('risk_assessment', {})
        strategy = analysis['agents_results'].get('strategy', {})
        payment = analysis['agents_results'].get('payment_prediction', {})
        compliance = analysis['agents_results'].get('compliance', {})
        segment = analysis['agents_results'].get('segmentation', {})

        # Build recommendation
        recommendation = {
            'action': strategy.get('recommended_strategy', 'standard_follow_up'),
            'channel': strategy.get(
                'recommended_channel', {}
            ).get('primary', 'email'),
            'timing': strategy.get('recommended_timing', {}),
            'expected_outcome': {
                'payment_probability': risk.get('payment_probability', 0),
                'expected_recovery': strategy.get('expected_recovery', {}),
            },
            'compliance_approved': compliance.get('is_approved', False),
            'debtor_segment': segment.get('segment_name', 'Unknown'),
        }

        # Narrative summary
        risk_level = risk.get('risk_level', 'unknown')
        pay_prob = risk.get('payment_probability', 0)
        action = recommendation['action']
        channel = recommendation['channel']

        summary_parts = [
            f"This debtor is classified as {risk_level} risk "
            f"with a {pay_prob:.0%} probability of payment.",
        ]

        if action == 'payment_plan_offer':
            summary_parts.append(
                "Recommend offering a structured payment plan."
            )
        elif action == 'settlement_offer':
            summary_parts.append(
                "Recommend offering a settlement at reduced balance."
            )
        elif action == 'soft_reminder':
            summary_parts.append(
                "A gentle reminder is the recommended approach."
            )
        elif action == 'escalation':
            summary_parts.append(
                "Consider escalation due to low responsiveness."
            )
        elif action == 'hardship_program':
            summary_parts.append(
                "Refer to hardship program based on debtor's situation."
            )
        else:
            summary_parts.append(f"Recommended strategy: {action}.")

        summary_parts.append(
            f"Use {channel} as the primary contact method."
        )

        if not compliance.get('is_approved', True):
            summary_parts.append(
                "⚠️ COMPLIANCE ISSUE: Action requires review "
                "before proceeding."
            )

        recommendation['summary'] = ' '.join(summary_parts)

        return recommendation

    def _calculate_priority(self, analysis: Dict) -> float:
        """Calculate a priority score for the debtor (0-100)"""
        risk = analysis['agents_results'].get('risk_assessment', {})
        payment = analysis['agents_results'].get('payment_prediction', {})
        compliance = analysis['agents_results'].get('compliance', {})

        # Components
        pay_prob = risk.get('payment_probability', 0)
        risk_score = risk.get('risk_score', 0.5)
        expected_value = payment.get('expected_value', 0)
        is_compliant = compliance.get('is_approved', True)

        # Priority formula
        priority = (
            pay_prob * 30 +  # Higher payment probability = higher priority
            (1 - risk_score) * 20 +  # Lower risk = higher priority
            min(expected_value * 100, 30) +  # Higher expected value
            (20 if is_compliant else 0)  # Compliance bonus
        )

        return round(min(max(priority, 0), 100), 1)

    def _update_agents_status(self):
        """Update status of all agents"""
        agents = {
            'risk_scoring': self.risk_agent,
            'communication': self.communication_agent,
            'payment_prediction': self.payment_agent,
            'strategy_optimization': self.strategy_agent,
            'compliance_monitoring': self.compliance_agent,
            'segmentation': self.segmentation_agent,
        }
        self.agents_status = {
            name: agent.get_status() for name, agent in agents.items()
        }

    def get_system_status(self) -> Dict:
        """Get complete system status"""
        self._update_agents_status()
        return {
            'system_initialized': self.is_initialized,
            'agents': self.agents_status,
            'data_stats': {
                'debtors_loaded': len(self.debtor_data) if hasattr(
                    self, 'debtor_data'
                ) else 0,
                'communications_loaded': len(self.comm_data) if hasattr(
                    self, 'comm_data'
                ) else 0,
            }
        }

    def interactive_demo(self):
        """Run an interactive demonstration of the system"""
        print("\n" + "=" * 70)
        print("   DEBT COLLECTION AI SYSTEM - INTERACTIVE DEMONSTRATION")
        print("=" * 70)

        # Pick sample debtors
        sample_debtors = self.debtor_data.sample(3)

        for _, debtor in sample_debtors.iterrows():
            debtor_id = debtor['debtor_id']
            print(f"\n{'─'*70}")
            print(f"📋 DEBTOR: {debtor['first_name']} {debtor['last_name']}")
            print(f"   ID: {debtor_id}")
            print(f"   Balance: ${debtor['remaining_balance']:,.2f}")
            print(f"   Days Past Due: {debtor['days_past_due']}")
            print(f"   Credit Score: {debtor['credit_score']}")
            print(f"   Status: {debtor['status']}")

            # Analyze with sample communication
            sample_texts = [
                "I want to pay but I need a payment plan",
                "Stop calling me, I know my rights",
                "Can we settle for a lower amount?"
            ]

            analysis = self.analyze_debtor(
                debtor_id=debtor_id,
                communication_text=np.random.choice(sample_texts)
            )

            if 'error' not in analysis:
                rec = analysis.get('final_recommendation', {})
                print(f"\n   🎯 RECOMMENDATION:")
                print(f"   {rec.get('summary', 'N/A')}")
                print(f"\n   Priority Score: {analysis.get('priority_score', 0)}/100")

                risk = analysis['agents_results'].get('risk_assessment', {})
                print(f"   Risk Level: {risk.get('risk_level', 'N/A')}")
                print(f"   Payment Probability: {risk.get('payment_probability', 0):.0%}")

                compliance = analysis['agents_results'].get('compliance', {})
                status = "✅ APPROVED" if compliance.get('is_approved') else "❌ BLOCKED"
                print(f"   Compliance: {status}")

                if analysis['agents_results'].get('generated_response'):
                    print(f"\n   📨 GENERATED RESPONSE (preview):")
                    response = analysis['agents_results'][
                        'generated_response'
                    ].get('response', '')
                    # Show first 200 chars
                    preview = response[:200] + "..." if len(response) > 200 else response
                    print(f"   {preview}")

        print(f"\n{'='*70}")
        print("   DEMONSTRATION COMPLETE")
        print(f"{'='*70}\n")


# =============================================================================
# SECTION 11: API / INTERFACE LAYER
# =============================================================================

class DebtCollectionAPI:
    """
    High-level API for the Debt Collection AI System.
    Provides clean interface for integration with existing systems.
    """

    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.initialized = False

    def setup(self, n_training_debtors: int = 1000,
              n_training_comms: int = 2000) -> Dict:
        """Initialize and train the complete system"""
        results = self.orchestrator.initialize_and_train(
            n_debtors=n_training_debtors,
            n_communications=n_training_comms
        )
        self.initialized = True
        return results

    def analyze_debtor(self, debtor_id: str = None,
                       profile: Dict = None,
                       message: str = None) -> Dict:
        """Analyze a single debtor"""
        if not self.initialized:
            return {'error': 'System not initialized. Call setup() first.'}
        return self.orchestrator.analyze_debtor(
            debtor_id=debtor_id,
            debtor_profile=profile,
            communication_text=message
        )

    def batch_analyze(self, debtor_ids: List[str] = None,
                      n: int = 20) -> List[Dict]:
        """Analyze multiple debtors"""
        if not self.initialized:
            return [{'error': 'System not initialized'}]
        return self.orchestrator.batch_analyze(debtor_ids, n)

    def analyze_message(self, text: str) -> Dict:
        """Analyze a debtor's message"""
        if not self.initialized:
            return {'error': 'System not initialized'}
        result = self.orchestrator.communication_agent.predict(text)
        return result.data if result.success else {'error': result.errors}

    def check_compliance(self, action: Dict, debtor: Dict) -> Dict:
        """Check an action for compliance"""
        result = self.orchestrator.compliance_agent.check_compliance(
            action, debtor
        )
        return result.data if result.success else {'error': result.errors}

    def get_status(self) -> Dict:
        """Get system status"""
        return self.orchestrator.get_system_status()

    def demo(self):
        """Run interactive demo"""
        if not self.initialized:
            print("Setting up system first...")
            self.setup()
        self.orchestrator.interactive_demo()


# =============================================================================
# SECTION 12: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function - demonstrates the complete system"""

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          MULTI-MODEL AI AGENTS FOR DEBT COLLECTION                   ║
║                                                                      ║
║   Comprehensive AI-Powered Debt Recovery System                      ║
║                                                                      ║
║   Agents:                                                            ║
║   1. Risk Scoring Agent (Ensemble ML)                                ║
║   2. Communication Agent (NLP + Deep Learning)                       ║
║   3. Payment Prediction Agent (Neural Networks)                      ║
║   4. Strategy Optimization Agent (Reinforcement Learning)            ║
║   5. Compliance Monitoring Agent (Rule-Based + NLP)                  ║
║   6. Debtor Segmentation Agent (Clustering)                          ║
║   7. Orchestrator Agent (Multi-Agent Coordination)                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize the system
    api = DebtCollectionAPI()

    print("🚀 Setting up and training the AI system...")
    print("   This will generate synthetic data and train all models.\n")

    setup_results = api.setup(
        n_training_debtors=1000,
        n_training_comms=2000
    )

    print(f"\n✅ System trained in "
          f"{setup_results['total_training_time_seconds']:.1f} seconds")

    # Display training results
    print("\n📊 TRAINING RESULTS:")
    print("─" * 50)

    if 'risk_agent' in setup_results:
        risk_res = setup_results['risk_agent']
        if 'ensemble' in risk_res:
            print(f"  Risk Scoring Agent:")
            print(f"    AUC: {risk_res['ensemble']['roc_auc']:.4f}")
            print(f"    F1:  {risk_res['ensemble']['f1_score']:.4f}")

    if 'communication_agent' in setup_results:
        comm_res = setup_results['communication_agent']
        if 'intent_classification' in comm_res:
            print(f"  Communication Agent:")
            print(f"    Intent Accuracy: "
                  f"{comm_res['intent_classification']['accuracy']:.4f}")
        if 'deep_sentiment' in comm_res:
            print(f"    Sentiment Accuracy: "
                  f"{comm_res['deep_sentiment']['accuracy']:.4f}")

    if 'payment_agent' in setup_results:
        pay_res = setup_results['payment_agent']
        if 'classifier' in pay_res:
            print(f"  Payment Prediction Agent:")
            print(f"    AUC: {pay_res['classifier']['roc_auc']:.4f}")

    if 'strategy_agent' in setup_results:
        strat_res = setup_results['strategy_agent']
        print(f"  Strategy Agent:")
        print(f"    Unique States: "
              f"{strat_res.get('unique_states', 'N/A')}")
        print(f"    Avg Reward: "
              f"{strat_res.get('avg_reward_last_1000', 'N/A')}")

    # Run interactive demo
    print("\n")
    api.demo()

    # Demonstrate individual capabilities
    print("\n📱 INDIVIDUAL AGENT DEMONSTRATIONS:")
    print("=" * 50)

    # 1. Message Analysis
    print("\n💬 Message Analysis Examples:")
    print("─" * 40)

    test_messages = [
        "I just got a new job and I'm ready to start paying this off. "
        "Can we set up a monthly plan?",

        "I'm recording this call. You have called me 5 times today. "
        "I'm filing a complaint with the CFPB. Get a lawyer ready.",

        "I lost my job three months ago and my wife is in the hospital. "
        "I want to pay but I literally cannot afford anything right now.",

        "What's the lowest amount you'll accept to settle this today? "
        "I have $2,000 available right now.",
    ]

    for msg in test_messages:
        print(f"\n  Message: \"{msg[:80]}...\"")
        result = api.analyze_message(msg)

        if 'vader_sentiment' in result:
            vader = result['vader_sentiment']
            print(f"  Sentiment: {result.get('vader_category', 'N/A')} "
                  f"(compound: {vader['compound']:.3f})")

        if 'intent' in result:
            intent = result['intent']
            print(f"  Intent: {intent['classification']} "
                  f"(confidence: {intent['confidence']:.2%})")

        if 'risk_indicators' in result:
            risks = result['risk_indicators']
            active_risks = [k for k, v in risks.items() if v]
            if active_risks:
                print(f"  ⚠️ Risk Indicators: {', '.join(active_risks)}")

        if 'compliance_flags' in result:
            flags = result['compliance_flags']
            if flags:
                print(f"  🚨 Compliance Flags: {', '.join(flags)}")

    # 2. Compliance Check Examples
    print("\n\n⚖️ Compliance Check Examples:")
    print("─" * 40)

    compliance_scenarios = [
        {
            'action': {
                'channel': 'phone',
                'time': '22:00',
                'message': 'Pay now or we will arrest you!'
            },
            'debtor': {
                'debtor_id': 'TEST-001',
                'do_not_call': False,
                'cease_and_desist': False,
                'represented_by_attorney': False,
                'bankruptcy_filed': False,
                'active_military': False,
                'contacts_today': 0,
                'contacts_this_week': 3,
            },
            'scenario_name': 'After-hours call with threatening language',
        },
        {
            'action': {
                'channel': 'phone',
                'time': '10:00',
                'message': 'Calling regarding your account.'
            },
            'debtor': {
                'debtor_id': 'TEST-002',
                'do_not_call': False,
                'cease_and_desist': True,
                'represented_by_attorney': True,
                'bankruptcy_filed': False,
                'active_military': False,
                'contacts_today': 0,
                'contacts_this_week': 1,
            },
            'scenario_name': 'Call to debtor with attorney & cease/desist',
        },
        {
            'action': {
                'channel': 'email',
                'time': '10:00',
                'message': (
                    'This is an attempt to collect a debt. '
                    'Please contact us to discuss payment options.'
                ),
            },
            'debtor': {
                'debtor_id': 'TEST-003',
                'do_not_call': False,
                'cease_and_desist': False,
                'represented_by_attorney': False,
                'bankruptcy_filed': False,
                'active_military': True,
                'contacts_today': 0,
                'contacts_this_week': 2,
            },
            'scenario_name': 'Compliant email to military member',
        },
    ]

    for scenario in compliance_scenarios:
        print(f"\n  Scenario: {scenario['scenario_name']}")
        result = api.check_compliance(
            scenario['action'], scenario['debtor']
        )
        status = result.get('compliance_status', 'unknown')
        status_emoji = {
            'compliant': '✅',
            'warning': '⚠️',
            'violation': '❌',
            'review_required': '🔍'
        }.get(status, '❓')

        print(f"  Status: {status_emoji} {status.upper()}")
        print(f"  Violations: {result.get('total_violations', 0)}")
        print(f"  Warnings: {result.get('total_warnings', 0)}")
        print(f"  Recommendation: {result.get('recommendation', 'N/A')}")

        if result.get('violations'):
            for v in result['violations'][:2]:
                print(f"    ❌ {v['rule']}: {v['description'][:80]}")

    # 3. Batch Analysis Summary
    print("\n\n📊 BATCH ANALYSIS (Top 10 Priority Debtors):")
    print("─" * 70)

    batch_results = api.batch_analyze(n_sample=10)

    print(f"{'ID':<15} {'Name':<20} {'Balance':>12} {'Risk':>10} "
          f"{'Strategy':<25} {'Priority':>8}")
    print("─" * 95)

    for result in batch_results[:10]:
        if 'error' in result:
            continue
        did = result.get('debtor_id', 'N/A')
        risk = result.get('agents_results', {}).get(
            'risk_assessment', {}
        )
        strategy = result.get('agents_results', {}).get(
            'strategy', {}
        )

        # Get name from data
        mask = api.orchestrator.debtor_data['debtor_id'] == did
        if mask.any():
            row = api.orchestrator.debtor_data[mask].iloc[0]
            name = f"{row['first_name']} {row['last_name']}"
            balance = row['remaining_balance']
        else:
            name = "Unknown"
            balance = 0

        print(
            f"{did:<15} {name:<20} ${balance:>10,.2f} "
            f"{risk.get('risk_level', 'N/A'):>10} "
            f"{strategy.get('recommended_strategy', 'N/A'):<25} "
            f"{result.get('priority_score', 0):>7.1f}"
        )

    # System status
    print("\n\n🔍 SYSTEM STATUS:")
    print("─" * 40)
    status = api.get_status()
    print(f"  System Initialized: {status['system_initialized']}")
    print(f"  Debtors Loaded: {status['data_stats']['debtors_loaded']}")
    print(f"  Communications Loaded: "
          f"{status['data_stats']['communications_loaded']}")
    print(f"\n  Agent Status:")
    for agent_name, agent_status in status['agents'].items():
        trained = "✅" if agent_status['is_trained'] else "❌"
        print(f"    {trained} {agent_name} v{agent_status['version']}")

    print("\n" + "=" * 70)
    print("  SYSTEM DEMONSTRATION COMPLETE")
    print("  All agents trained and operational.")
    print("=" * 70 + "\n")

    return api


if __name__ == "__main__":
    api = main()

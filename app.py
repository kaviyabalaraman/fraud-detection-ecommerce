"""
E-Commerce Fraud Detection — Streamlit Application
Run with:  streamlit run app.py
"""

import pickle
import numpy as np
import streamlit as st

import streamlit as st

st.write("App started successfully 🚀")

# ──────────────────────────────────────────────────────────────────
# Page configuration (MUST be first Streamlit call)
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard · E-Commerce Fraud Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────
# Custom CSS — dark industrial theme with amber accents
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ──────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Global reset ──────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d0d;
    color: #e8e8e8;
}

/* ── Main container ────────────────────────────────── */
.block-container {
    max-width: 760px !important;
    padding: 2.5rem 2rem 4rem !important;
}

/* ── Header strip ──────────────────────────────────── */
.header-strip {
    border-top: 3px solid #f5a623;
    padding: 2rem 0 1.2rem;
    margin-bottom: 0.5rem;
}
.header-strip h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.5rem;
    color: #ffffff;
    letter-spacing: -1px;
    margin: 0 0 0.3rem;
    line-height: 1.1;
}
.header-strip h1 span { color: #f5a623; }
.header-strip p {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #888;
    margin: 0;
    letter-spacing: 0.5px;
}

/* ── Section label ─────────────────────────────────── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #f5a623;
    margin: 2rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e1e1e;
}

/* ── Inputs ────────────────────────────────────────── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background-color: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #e8e8e8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.88rem !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #f5a623 !important;
    box-shadow: 0 0 0 2px rgba(245,166,35,0.15) !important;
}

/* ── Labels ────────────────────────────────────────── */
label, .stSelectbox label, .stNumberInput label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #aaaaaa !important;
}

/* ── Button ────────────────────────────────────────── */
div[data-testid="stButton"] > button {
    background-color: #f5a623 !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2.5rem !important;
    margin-top: 1.2rem !important;
    transition: background-color 0.15s ease, transform 0.1s ease !important;
    cursor: pointer !important;
}
div[data-testid="stButton"] > button:hover {
    background-color: #ffc04d !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Result cards ──────────────────────────────────── */
.result-card {
    margin-top: 2rem;
    padding: 2rem 2rem 1.8rem;
    border-radius: 4px;
    border-left: 4px solid;
}
.result-card.fraud {
    background: #1a0a0a;
    border-color: #e53935;
}
.result-card.safe {
    background: #0a150a;
    border-color: #43a047;
}
.result-card .verdict {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.55rem;
    margin: 0 0 0.5rem;
    line-height: 1.2;
}
.result-card.fraud .verdict { color: #ff5252; }
.result-card.safe  .verdict { color: #66bb6a; }
.result-card .sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.73rem;
    letter-spacing: 1px;
    color: #777;
}

/* ── Probability bar ───────────────────────────────── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 1.2rem;
}
.prob-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #777;
    min-width: 90px;
}
.prob-track {
    flex: 1;
    height: 6px;
    background: #1e1e1e;
    border-radius: 3px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}
.prob-fill.fraud-fill { background: #e53935; }
.prob-fill.safe-fill  { background: #43a047; }
.prob-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #aaa;
    min-width: 42px;
    text-align: right;
}

/* ── Divider ───────────────────────────────────────── */
hr { border-color: #1e1e1e !important; }

/* ── Feature importance table ──────────────────────── */
.fi-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.fi-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #999;
    min-width: 160px;
}
.fi-bar-bg {
    flex: 1;
    height: 4px;
    background: #1a1a1a;
    border-radius: 2px;
}
.fi-bar { height: 100%; background: #f5a623; border-radius: 2px; }
.fi-val {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #666;
    min-width: 40px;
    text-align: right;
}

/* ── Footer ────────────────────────────────────────── */
.footer {
    margin-top: 4rem;
    padding-top: 1.2rem;
    border-top: 1px solid #1a1a1a;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #444;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# Load model (cached)
# ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "⚠️  **Model file not found.**  "
        "Please run `python train_model.py` first to generate `fraud_model.pkl`."
    )
    st.stop()


# ──────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-strip">
  <h1>Fraud<span>Guard</span></h1>
  <p>E-COMMERCE TRANSACTION FRAUD DETECTOR · RANDOM FOREST CLASSIFIER · v1.0</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Enter the details of a transaction below and click **ANALYSE** to instantly "
    "predict whether it is legitimate or potentially fraudulent."
)


# ──────────────────────────────────────────────────────────────────
# Input form
# ──────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Transaction Details</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    transaction_amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=1.0,
        max_value=100_000.0,
        value=250.0,
        step=10.0,
        help="Total value of the transaction in INR",
    )

with col2:
    account_age_days = st.number_input(
        "Account Age (days)",
        min_value=1,
        max_value=5000,
        value=365,
        step=1,
        help="Number of days since the account was created",
    )

st.markdown('<p class="section-label">Payment & Device Info</p>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

# Encoding maps (must match training)
PAYMENT_MAP = {
    "Credit Card":  0,
    "Debit Card":   1,
    "UPI":          2,
    "Net Banking":  3,
}
DEVICE_MAP = {
    "Mobile":  0,
    "Desktop": 1,
}

with col3:
    payment_method_str = st.selectbox(
        "Payment Method",
        options=list(PAYMENT_MAP.keys()),
        index=0,
    )

with col4:
    device_used_str = st.selectbox(
        "Device Used",
        options=list(DEVICE_MAP.keys()),
        index=0,
    )

# ──────────────────────────────────────────────────────────────────
# Predict button
# ──────────────────────────────────────────────────────────────────
predict_btn = st.button("🔍  ANALYSE TRANSACTION")

if predict_btn:
    # ── Encode inputs ──────────────────────────────────────────────
    payment_encoded = PAYMENT_MAP[payment_method_str]
    device_encoded  = DEVICE_MAP[device_used_str]

    import pandas as pd
    features = pd.DataFrame([[
        transaction_amount,
        account_age_days,
        payment_encoded,
        device_encoded,
    ]], columns=["transaction_amount", "account_age_days", "payment_method", "device_used"])

    # ── Predict ────────────────────────────────────────────────────
    prediction   = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]   # [P(not fraud), P(fraud)]
    fraud_prob   = probabilities[1]
    safe_prob    = probabilities[0]

    # ── Result card ────────────────────────────────────────────────
    if prediction == 1:
        st.markdown(f"""
        <div class="result-card fraud">
          <div class="verdict">🚨 Fraudulent Transaction</div>
          <div class="sub">HIGH RISK · TRANSACTION FLAGGED FOR REVIEW</div>
          <div class="prob-row">
            <span class="prob-label">Fraud Risk</span>
            <div class="prob-track">
              <div class="prob-fill fraud-fill" style="width:{fraud_prob*100:.1f}%"></div>
            </div>
            <span class="prob-pct">{fraud_prob*100:.1f}%</span>
          </div>
          <div class="prob-row">
            <span class="prob-label">Legitimate</span>
            <div class="prob-track">
              <div class="prob-fill safe-fill" style="width:{safe_prob*100:.1f}%"></div>
            </div>
            <span class="prob-pct">{safe_prob*100:.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card safe">
          <div class="verdict">✅ Normal Transaction</div>
          <div class="sub">LOW RISK · NO SUSPICIOUS ACTIVITY DETECTED</div>
          <div class="prob-row">
            <span class="prob-label">Legitimate</span>
            <div class="prob-track">
              <div class="prob-fill safe-fill" style="width:{safe_prob*100:.1f}%"></div>
            </div>
            <span class="prob-pct">{safe_prob*100:.1f}%</span>
          </div>
          <div class="prob-row">
            <span class="prob-label">Fraud Risk</span>
            <div class="prob-track">
              <div class="prob-fill fraud-fill" style="width:{fraud_prob*100:.1f}%"></div>
            </div>
            <span class="prob-pct">{fraud_prob*100:.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature importance mini-chart ──────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:2.5rem">Feature Importance</p>', unsafe_allow_html=True)

    feature_names = [
        "Transaction Amount",
        "Account Age (days)",
        "Payment Method",
        "Device Used",
    ]
    importances = model.feature_importances_
    max_imp     = importances.max()

    fi_html = ""
    for name, imp in zip(feature_names, importances):
        bar_pct = imp / max_imp * 100
        fi_html += f"""
        <div class="fi-row">
          <span class="fi-name">{name}</span>
          <div class="fi-bar-bg">
            <div class="fi-bar" style="width:{bar_pct:.1f}%"></div>
          </div>
          <span class="fi-val">{imp:.3f}</span>
        </div>"""

    st.markdown(fi_html, unsafe_allow_html=True)

    # ── Input echo ─────────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:2.5rem">Input Summary</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amount",         f"₹{transaction_amount:,.0f}")
    c2.metric("Account Age",    f"{account_age_days} days")
    c3.metric("Payment",        payment_method_str)
    c4.metric("Device",         device_used_str)


# ──────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  FRAUDGUARD · RANDOM FOREST CLASSIFIER · ACADEMIC PROJECT DEMO<br>
  NOT FOR PRODUCTION USE · PREDICTIONS ARE BASED ON SYNTHETIC TRAINING DATA
</div>
""", unsafe_allow_html=True)
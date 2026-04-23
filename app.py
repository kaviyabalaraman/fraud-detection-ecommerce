import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load columns
columns = pickle.load(open('columns.pkl', 'rb'))

# Page config
st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🛡️ FraudGuard")
st.write("E-commerce Fraud Detection System")

# Inputs
col1, col2 = st.columns(2)

with col1:
    transaction_amount = st.number_input("Transaction Amount (₹)", value=250.0)

with col2:
    account_age_days = st.number_input("Account Age (Days)", value=365)

payment_method_str = st.selectbox(
    "Payment Method",
    ["Credit Card", "Debit Card", "UPI", "Net Banking"]
)

device_used_str = st.selectbox(
    "Device Used",
    ["Mobile", "Desktop"]
)

# Button
predict_btn = st.button("🔍 ANALYSE TRANSACTION")

if predict_btn:

    # Create input structure
    input_dict = dict.fromkeys(columns, 0)

    # Fill numeric values
    input_dict['Transaction Amount'] = transaction_amount
    input_dict['Account Age Days'] = account_age_days

    # Payment encoding
    if payment_method_str == "Credit Card":
        input_dict['Payment Method_credit card'] = 1
    elif payment_method_str == "Debit Card":
        input_dict['Payment Method_debit card'] = 1
    elif payment_method_str == "UPI":
        input_dict['Payment Method_upi'] = 1
    elif payment_method_str == "Net Banking":
        input_dict['Payment Method_net banking'] = 1

    # Device encoding
    if device_used_str == "Mobile":
        input_dict['Device Used_mobile'] = 1
    elif device_used_str == "Desktop":
        input_dict['Device Used_desktop'] = 1

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # Prediction
    prediction = model.predict(input_df)[0]

    # Output
    if prediction == 1:
        st.error("🚨 Fraud Transaction Detected")
    else:
        st.success("✅ Normal Transaction")

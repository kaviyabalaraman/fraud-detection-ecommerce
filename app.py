import pickle
import pandas as pd
import streamlit as st

# Load columns
columns = pickle.load(open('columns.pkl', 'rb'))

# Load model
@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# UI
st.title("🛡️ FraudGuard")
st.write("E-commerce Fraud Detection")

# Inputs
col1, col2 = st.columns(2)

with col1:
    transaction_amount = st.number_input("Transaction Amount", value=250.0)

with col2:
    account_age_days = st.number_input("Account Age Days", value=365)

payment_method_str = st.selectbox(
    "Payment Method",
    ["Credit Card", "Debit Card", "UPI", "Net Banking"]
)

device_used_str = st.selectbox(
    "Device Used",
    ["Mobile", "Desktop"]
)

# Button
if st.button("🔍 ANALYSE TRANSACTION"):

    # Create input with all columns = 0
    input_dict = {col: 0 for col in columns}

    # Fill numeric safely
    for col in columns:
        if "transaction" in col.lower():
            input_dict[col] = transaction_amount
        if "account age" in col.lower():
            input_dict[col] = account_age_days

    # Fill payment dynamically
    for col in columns:
        if payment_method_str.lower() in col.lower():
            input_dict[col] = 1

    # Fill device dynamically
    for col in columns:
        if device_used_str.lower() in col.lower():
            input_dict[col] = 1

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)[0]

    # Output
    if prediction == 1:
        st.error("🚨 Fraud Transaction")
    else:
        st.success("✅ Normal Transaction")

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

   # Create dataframe with correct columns
input_df = pd.DataFrame(columns=columns)
input_df.loc[0] = 0

# Fill numeric values (ONLY if column exists)
if 'Transaction Amount' in input_df.columns:
    input_df.at[0, 'Transaction Amount'] = transaction_amount

if 'Account Age Days' in input_df.columns:
    input_df.at[0, 'Account Age Days'] = account_age_days

# Payment encoding
for col in input_df.columns:
    if col.lower().startswith('payment method'):
        if payment_method_str.lower() in col.lower():
            input_df.at[0, col] = 1

# Device encoding
for col in input_df.columns:
    if col.lower().startswith('device used'):
        if device_used_str.lower() in col.lower():
            input_df.at[0, col] = 1

# FINAL SAFETY (very important)
input_df = input_df[columns]

# Predict
prediction = model.predict(input_df)[0]

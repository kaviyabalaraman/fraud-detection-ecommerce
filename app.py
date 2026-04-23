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

if st.button("🔍 ANALYSE TRANSACTION"):

    # Step 1: Create empty dataframe with exact columns
    input_df = pd.DataFrame([0]*len(columns)).T
    input_df.columns = columns

    # Step 2: Fill ONLY known fields (safe)
    try:
        input_df.loc[0, 'Transaction Amount'] = transaction_amount
    except:
        pass

    try:
        input_df.loc[0, 'Account Age Days'] = account_age_days
    except:
        pass

    # Step 3: Payment encoding
    for col in columns:
        if "payment" in col.lower() and payment_method_str.lower() in col.lower():
            input_df.loc[0, col] = 1

    # Step 4: Device encoding
    for col in columns:
        if "device" in col.lower() and device_used_str.lower() in col.lower():
            input_df.loc[0, col] = 1

    # Step 5: Ensure correct order (CRITICAL)
    input_df = input_df.reindex(columns=columns)

    # Step 6: Predict
    prediction = model.predict(input_df)[0]

    # Step 7: Output
    if prediction == 1:
        st.error("🚨 Fraud Transaction")
    else:
        st.success("✅ Normal Transaction")


st.write(columns)

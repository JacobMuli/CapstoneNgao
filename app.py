import streamlit as st
import pandas as pd
import joblib
import requests
import os

RAW_BASE = "https://raw.githubusercontent.com/JacobMuli/CapstoneNgao/main/"

def download_file(filename):
    url = RAW_BASE + filename
    r = requests.get(url)
    if r.status_code == 200:
        with open(filename, "wb") as f:
            f.write(r.content)
    else:
        st.error(f"Failed to download {filename}")

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        download_file("model.pkl")
    if not os.path.exists("encoders.pkl"):
        download_file("encoders.pkl")
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

# ---------------------------------------------------

st.title("Customer Churn Prediction (Random Forest Model)")

model, encoders = load_model()

categorical_cols = ["Gender", "Subscription Type", "Contract Length"]
numeric_cols = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction"
]

# Dropdown options
gender_options = ["Female", "Male"]
subscription_options = ["Standard", "Basic", "Premium"]
contract_options = ["Annual", "Monthly", "Quarterly"]

inputs = {}

st.subheader("Enter Customer Information")

# Categorical UI
inputs["Gender"] = st.selectbox("Gender", gender_options)
inputs["Subscription Type"] = st.selectbox("Subscription Type", subscription_options)
inputs["Contract Length"] = st.selectbox("Contract Length", contract_options)

# Numeric UI (integer only)
label_map = {
    "Age": "Age (years)",
    "Tenure": "Tenure (months)",
    "Usage Frequency": "Usage Frequency (times/month)",
    "Support Calls": "Customer Support Calls (count)",
    "Payment Delay": "Payment Delay (days)",
    "Total Spend": "Total Spend (KSh)",
    "Last Interaction": "Days Since Last Interaction"
}

for col in numeric_cols:
    inputs[col] = st.number_input(label_map[col], min_value=0, step=1)

# Prediction
if st.button("Predict"):

    df = pd.DataFrame([inputs])

    # Encode categoricals
    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col].astype(str))

    prediction = model.predict(df)[0]
    result = "Churn" if prediction == 1 else "No Churn"

    st.success(f"Prediction: **{result}**")


import streamlit as st
import pandas as pd
import joblib
import requests
import os

# =========================================================
#  GitHub RAW base URL for downloading model + encoders
# =========================================================
RAW_BASE = "https://raw.githubusercontent.com/JacobMuli/CapstoneNgao/main/"

def download_file(filename):
    url = RAW_BASE + filename
    r = requests.get(url)
    if r.status_code == 200:
        with open(filename, "wb") as f:
            f.write(r.content)
    else:
        st.error(f"Failed to download: {filename}. Check URL: {url}")

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        download_file("model.pkl")
    if not os.path.exists("encoders.pkl"):
        download_file("encoders.pkl")

    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

# =========================================================
#  Streamlit App
# =========================================================
st.title("Customer Churn Prediction (XGBoost Model)")

model, encoders = load_model()

# --------------------------
# FIXED CATEGORY OPTIONS
# --------------------------
gender_options = ["Female", "Male"]
subscription_options = ["Standard", "Basic", "Premium"]
contract_options = ["Annual", "Monthly", "Quarterly"]

# --------------------------
# FIXED FEATURE LISTS
# --------------------------
categorical_cols = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

numeric_cols = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction"
]

# --------------------------
# USER INPUT UI
# --------------------------
st.subheader("Enter Customer Information")

inputs = {}

# Categorical Inputs
inputs["Gender"] = st.selectbox("Gender", gender_options)
inputs["Subscription Type"] = st.selectbox("Subscription Type", subscription_options)
inputs["Contract Length"] = st.selectbox("Contract Length", contract_options)

# Numeric Inputs (Integers Only)
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

# --------------------------
# PREDICT
# --------------------------
if st.button("Predict"):

    df = pd.DataFrame([inputs])

    # Apply encoders
    for col, enc in encoders.items():
        if col != "Churn":
            df[col] = enc.transform(df[col].astype(str))

    # Ensure numeric columns are int (to match training)
    for col in numeric_cols:
        df[col] = df[col].astype(int)

    # Reorder EXACTLY as model expects
    feature_order = [
        'Age',
        'Gender',
        'Tenure',
        'Usage Frequency',
        'Support Calls',
        'Payment Delay',
        'Subscription Type',
        'Contract Length',
        'Total Spend',
        'Last Interaction'
    ]

    df = df[feature_order]

    # Predict
    prediction = model.predict(df)[0]
    result = "Churn" if prediction == 1 else "No Churn"

    st.success(f"Prediction: **{result}**")

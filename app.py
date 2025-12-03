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
        st.error(f"Failed to download: {filename}")

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        download_file("model.pkl")
    if not os.path.exists("encoders.pkl"):
        download_file("encoders.pkl")

    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

st.title("Customer Churn Prediction (XGBoost Model)")

model, encoders = load_model()

# ---- Dropdown options ----
gender_options = ["Female", "Male"]
subscription_options = ["Standard", "Basic", "Premium"]
contract_options = ["Annual", "Monthly", "Quarterly"]

# ---- Identify model feature columns ----
categorical_cols = [c for c in encoders.keys() if c != "Churn"]
numeric_cols = [c for c in model.get_booster().feature_names if c not in categorical_cols]

# Remove CustomerID if present
if "CustomerID" in categorical_cols:
    categorical_cols.remove("CustomerID")
if "CustomerID" in numeric_cols:
    numeric_cols.remove("CustomerID")

st.subheader("Enter Customer Information")

inputs = {}

# ---- Categorical Inputs ----
for col in categorical_cols:
    if col == "Gender":
        inputs[col] = st.selectbox("Gender", gender_options)
    elif col == "Subscription Type":
        inputs[col] = st.selectbox("Subscription Type", subscription_options)
    elif col == "Contract Length":
        inputs[col] = st.selectbox("Contract Length", contract_options)
    else:
        inputs[col] = st.text_input(col)

# ---- Numeric Inputs (Integers Only) ----
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
    label = label_map.get(col, col)
    inputs[col] = st.number_input(label, min_value=0, step=1)

# ---- Predict ----
if st.button("Predict"):
    df = pd.DataFrame([inputs])

    # Apply encoders to categorical fields
    for col, enc in encoders.items():
        if col != "Churn":
            df[col] = enc.transform(df[col].astype(str))

    # Reorder columns to EXACT model training order
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

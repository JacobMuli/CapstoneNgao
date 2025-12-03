import streamlit as st
import pandas as pd
import joblib
import requests
import os

RAW_BASE = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/"

def download_file(filename):
    url = RAW_BASE + filename
    r = requests.get(url)
    if r.status_code == 200:
        with open(filename, "wb") as f:
            f.write(r.content)
    else:
        st.error(f"Failed to download {filename} from GitHub")

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

categorical_cols = [c for c in encoders.keys() if c != "Churn"]
numeric_cols = [c for c in model.get_booster().feature_names if c not in categorical_cols]

st.subheader("Enter Customer Information")

inputs = {}

# Categorical fields
for col in categorical_cols:
    val = st.text_input(col, "")
    inputs[col] = val

# Numeric fields
for col in numeric_cols:
    val = st.number_input(col, value=0.0)
    inputs[col] = val

if st.button("Predict"):
    df = pd.DataFrame([inputs])

    # Apply encoders
    for col, enc in encoders.items():
        if col != "Churn":
            df[col] = enc.transform(df[col].astype(str))

    prediction = model.predict(df)[0]
    result = "Churn" if prediction == 1 else "No Churn"

    st.success(f"Prediction: **{result}**")


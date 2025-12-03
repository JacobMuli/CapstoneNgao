# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import tensorflow as tf
import json

# Paths (use your uploaded files)
PREPROCESSOR_PATH = Path("/mnt/data/preprocessor.joblib")
TRAIN_COLUMNS_PATH = Path("/mnt/data/train_columns.joblib")
BEST_MODEL_PATH = Path("/mnt/data/best_model.h5")
FINAL_MODEL_PATH = Path("/mnt/data/final_model.h5")
TRAIN_CSV_PATH = Path("/mnt/data/customer_churn_dataset-training-master.csv")  # optional sample

@st.cache_resource
def load_artifacts():
    # Load preprocessor
    pre = joblib.load(PREPROCESSOR_PATH)
    train_cols = joblib.load(TRAIN_COLUMNS_PATH)
    # Load keras model: prefer best_model then final_model
    model_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else FINAL_MODEL_PATH
    model = None
    if model_path.exists():
        model = tf.keras.models.load_model(str(model_path))
    else:
        st.warning(f"No Keras model found at {BEST_MODEL_PATH} or {FINAL_MODEL_PATH}.")
    # try to load optional label encoder
    le = None
    try:
        le = joblib.load(PREPROCESSOR_PATH.parent / "label_encoder.joblib")
    except Exception:
        pass
    return pre, train_cols, model, le

preprocessor, train_columns, model, label_encoder = load_artifacts()

st.title("Churn Prediction — Neural Network")

st.markdown(
    """
    **Input options**
    - Paste a JSON object with feature names matching training columns (preferred)
    - Or click "Load sample (first row)" to inspect training sample
    """
)

# Button to load first training row (if CSV exists)
if st.button("Load sample (first row)"):
    try:
        if TRAIN_CSV_PATH.exists():
            sample = pd.read_csv(TRAIN_CSV_PATH).drop(columns=[col for col in [*pd.read_csv(TRAIN_CSV_PATH).columns] if col.lower() == 'churn' or col.lower() == 'target' or col.lower() == 'label'], errors='ignore')
            st.dataframe(sample.head(1))
        else:
            st.warning("Training CSV not found at /mnt/data/customer_churn_dataset-training-master.csv")
    except Exception as e:
        st.error(f"Failed to load sample: {e}")

st.subheader("Option A — Paste JSON for a single row")
input_json = st.text_area("Feature JSON (single row)", value='{}', height=180)
if st.button("Predict from JSON"):
    try:
        # parse JSON into 1-row DataFrame
        row = pd.read_json(input_json, typ='series')
        df = pd.DataFrame([row])
        # Ensure columns match training columns: add missing cols with NaN
        for c in train_columns:
            if c not in df.columns:
                df[c] = np.nan
        # Keep only training columns (same order)
        df = df[train_columns]
        # Preprocess
        X_proc = preprocessor.transform(df)
        proba = float(model.predict(X_proc).ravel()[0])
        pred = int(proba >= 0.5)
        label = label_encoder.inverse_transform([pred])[0] if label_encoder is not None else pred
        st.metric("Predicted probability of churn", f"{proba:.4f}")
        st.write("Predicted label:", label)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.subheader("Option B — Paste CSV (single row) or comma-separated features")
csv_text = st.text_area("Or paste a CSV row (header optional)", value="", height=120)
if st.button("Predict from CSV row"):
    try:
        if csv_text.strip() == "":
            st.warning("Paste a CSV row or disable this option.")
        else:
            # try reading; if header present pandas will handle it
            from io import StringIO
            df_try = pd.read_csv(StringIO(csv_text))
            if df_try.shape[0] > 1:
                df_try = df_try.head(1)
            # align columns
            for c in train_columns:
                if c not in df_try.columns:
                    df_try[c] = np.nan
            df_try = df_try[train_columns]
            X_proc = preprocessor.transform(df_try)
            proba = float(model.predict(X_proc).ravel()[0])
            pred = int(proba >= 0.5)
            label = label_encoder.inverse_transform([pred])[0] if label_encoder is not None else pred
            st.metric("Predicted probability of churn", f"{proba:.4f}")
            st.write("Predicted label:", label)
    except Exception as e:
        st.error(f"CSV prediction failed: {e}")

st.markdown("---")
st.write("App expects the preprocessor and Keras model saved to the given `/mnt/data` paths.")
st.write("To run locally: `pip install -r requirements.txt` then `streamlit run app.py`.")

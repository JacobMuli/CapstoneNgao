# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import tensorflow as tf
import json

# Candidate paths (repo root first, then fallback)
APP_ROOT = Path(__file__).parent.resolve()
FALLBACK_DIR = Path("/mnt/data")

CANDIDATE_DIRS = [APP_ROOT, FALLBACK_DIR]

PREPROCESSOR_NAME = "preprocessor.joblib"
TRAIN_COLUMNS_NAME = "train_columns.joblib"
BEST_MODEL_NAME = "best_model.h5"
FINAL_MODEL_NAME = "final_model.h5"
LABEL_ENCODER_NAME = "label_encoder.joblib"
TRAIN_CSV_NAME = "customer_churn_dataset-training-master.csv"  # optional sample

def find_file_in_candidates(fname):
    for d in CANDIDATE_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None

# Sidebar snapshot for debugging
st.sidebar.markdown("### Files in candidate folders")
for d in CANDIDATE_DIRS:
    try:
        entries = sorted([p.name for p in d.iterdir()]) if d.exists() else []
        st.sidebar.markdown(f"**{d}** ({len(entries)})")
        for e in entries[:30]:
            st.sidebar.text(e)
        if len(entries) > 30:
            st.sidebar.text("... (truncated)")
    except Exception as ex:
        st.sidebar.text(f"cannot list {d}: {ex}")

@st.cache_resource
def load_artifacts():
    # find preprocessor & train_columns
    pre_path = find_file_in_candidates(PREPROCESSOR_NAME)
    cols_path = find_file_in_candidates(TRAIN_COLUMNS_NAME)
    if pre_path is None or cols_path is None:
        raise RuntimeError(f"Missing artifacts. Expected {PREPROCESSOR_NAME} and {TRAIN_COLUMNS_NAME} in repo root or /mnt/data.")

    pre = joblib.load(pre_path)
    train_cols = joblib.load(cols_path)

    # find model (prefer best_model then final_model)
    best_path = find_file_in_candidates(BEST_MODEL_NAME)
    final_path = find_file_in_candidates(FINAL_MODEL_NAME)
    model_path = best_path if best_path is not None else final_path
    model = None
    if model_path is not None:
        model = tf.keras.models.load_model(str(model_path))
    else:
        st.warning("No Keras model found (best_model.h5 or final_model.h5) in repo root or /mnt/data.")

    # optional label encoder
    le_path = find_file_in_candidates(LABEL_ENCODER_NAME)
    le = None
    if le_path is not None:
        try:
            le = joblib.load(le_path)
        except Exception:
            le = None

    return pre, train_cols, model, le

# Load artifacts
try:
    preprocessor, train_columns, model, label_encoder = load_artifacts()
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

st.title("Churn Prediction — Neural Network")

st.markdown(
    """
    **Input options**
    - Paste a JSON object with feature names matching training columns (preferred)
    - Or click "Load sample (first row)" to inspect training sample
    """
)

# Button to load first training row (if CSV exists in candidates)
if st.button("Load sample (first row)"):
    # try candidate locations for the CSV
    csv_path = find_file_in_candidates(TRAIN_CSV_NAME)
    if csv_path is None:
        st.warning(f"Training CSV not found ({TRAIN_CSV_NAME}) in repo root or /mnt/data")
    else:
        try:
            sample_df = pd.read_csv(csv_path)
            # drop common target-like columns if present
            drop_cols = [c for c in sample_df.columns if c.lower() in ("churn","target","label")]
            sample_df = sample_df.drop(columns=drop_cols, errors='ignore')
            st.dataframe(sample_df.head(1))
        except Exception as ex:
            st.error(f"Failed to load sample CSV: {ex}")

st.subheader("Option A — Paste JSON for a single row")
input_json = st.text_area("Feature JSON (single row)", value='{}', height=180)
if st.button("Predict from JSON"):
    try:
        row = pd.read_json(input_json, typ='series')
        df = pd.DataFrame([row])
        # Ensure columns match training columns: add missing cols with NaN
        for c in train_columns:
            if c not in df.columns:
                df[c] = np.nan
        df = df[train_columns]
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
            from io import StringIO
            df_try = pd.read_csv(StringIO(csv_text))
            if df_try.shape[0] > 1:
                df_try = df_try.head(1)
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
st.write("App expects artifacts saved in the repo root (or /mnt/data as fallback).")
st.write("To run locally: `pip install -r requirements.txt` then `streamlit run app.py`.")

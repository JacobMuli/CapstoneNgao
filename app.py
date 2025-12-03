# app.py (patched for stable deterministic predictions)
import os
# Force CPU inference to reduce floating-point nondeterminism across runs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import tensorflow as tf
import json
import traceback
import ast

st.set_page_config(page_title="Churn Prediction — Stable NN", layout="wide")

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

# ---------- small util helpers ----------
def find_file_in_candidates(fname):
    for d in CANDIDATE_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None

def safe_title_cat(s):
    # convert to string, strip, title-case; keep empty strings as ""
    if pd.isna(s):
        return ""
    try:
        return str(s).strip().title()
    except Exception:
        return str(s).strip()

def safe_parse_json(text):
    """
    Tolerant parser: accepts valid JSON or Python dict notation.
    Returns a Python object (dict/list/etc) or raises an exception on failure.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty input")
    # If it looks like a Python dict (single quotes) try ast.literal_eval first
    try:
        # heuristic: starts with "{" and contains single quotes but not double quotes
        if text.startswith("{") and ("'" in text and '"' not in text):
            parsed_py = ast.literal_eval(text)
            # convert to JSON-serializable structure via json.dumps/load
            return json.loads(json.dumps(parsed_py))
    except Exception:
        # fall through to json.loads
        pass
    # Try json.loads (normal JSON)
    return json.loads(text)

# Sidebar snapshot for debugging
st.sidebar.markdown("### Files in candidate folders (debug)")
for d in CANDIDATE_DIRS:
    try:
        entries = sorted([p.name for p in d.iterdir()]) if d.exists() else []
        st.sidebar.markdown(f"**{d}** ({len(entries)})")
        for e in entries[:40]:
            st.sidebar.text(e)
        if len(entries) > 40:
            st.sidebar.text("... (truncated)")
    except Exception as ex:
        st.sidebar.text(f"cannot list {d}: {ex}")

# Toggle detailed debug output
DEBUG = st.sidebar.checkbox("Show debug info (processed vectors)", value=False)

@st.cache_resource
def load_artifacts():
    """
    Load preprocessor, train_columns, model, and optional label encoder.
    Also attempt to extract numeric_cols and cat_cols from fitted preprocessor.
    """
    # locate artifacts
    pre_path = find_file_in_candidates(PREPROCESSOR_NAME)
    cols_path = find_file_in_candidates(TRAIN_COLUMNS_NAME)

    if pre_path is None or cols_path is None:
        raise RuntimeError(f"Missing artifacts. Expected {PREPROCESSOR_NAME} and {TRAIN_COLUMNS_NAME} in repo root or /mnt/data.")

    pre = joblib.load(pre_path)
    train_cols = joblib.load(cols_path)

    # Extract numeric and categorical columns from ColumnTransformer if possible
    numeric_cols = []
    cat_cols = []
    try:
        # pre.transformers_ is available on fitted ColumnTransformer
        if hasattr(pre, "transformers_"):
            for name, trans, cols in pre.transformers_:
                if cols is None:
                    continue
                # name could be 'num'/'cat' or custom
                cname = str(name).lower()
                # treat known numeric transformer names or check for scaler in named_steps
                is_num = False
                is_cat = False
                try:
                    # some objects expose named_steps
                    if hasattr(trans, "named_steps") and "scaler" in trans.named_steps:
                        is_num = True
                    if hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
                        is_cat = True
                except Exception:
                    pass
                if "num" in cname or is_num:
                    numeric_cols = list(cols)
                if "cat" in cname or is_cat:
                    cat_cols = list(cols)
    except Exception:
        # fallback: leave lists empty and we will try to infer later from sample CSV if present
        numeric_cols = []
        cat_cols = []

    # find model (prefer best_model then final_model)
    best_path = find_file_in_candidates(BEST_MODEL_NAME)
    final_path = find_file_in_candidates(FINAL_MODEL_NAME)
    model_path = best_path if best_path is not None else final_path
    model = None
    if model_path is not None:
        model = tf.keras.models.load_model(str(model_path))
    else:
        raise RuntimeError("No Keras model found (best_model.h5 or final_model.h5) in repo root or /mnt/data.")

    # optional label encoder if saved
    le_path = find_file_in_candidates(LABEL_ENCODER_NAME)
    le = None
    if le_path is not None:
        try:
            le = joblib.load(le_path)
        except Exception:
            le = None

    return pre, train_cols, model, le, numeric_cols, cat_cols

# Load artifacts (stop app if fail)
try:
    preprocessor, train_columns, model, label_encoder, numeric_cols, cat_cols = load_artifacts()
except Exception as e:
    st.error("Failed to load model artifacts. See sidebar for file listing.")
    st.exception(e)
    st.stop()

# If cat/numeric columns not extracted, try to infer from training CSV (best-effort)
if not numeric_cols or not cat_cols:
    sample_csv = find_file_in_candidates(TRAIN_CSV_NAME)
    if sample_csv is not None:
        try:
            sample_df = pd.read_csv(sample_csv)
            # drop possible target columns heuristically
            drop_target = [c for c in sample_df.columns if c.lower() in ("churn", "target", "label", "exited")]
            for c in drop_target:
                sample_df = sample_df.drop(columns=[c], errors='ignore')
            inferred_num = sample_df.select_dtypes(include=["int64","float64"]).columns.tolist()
            inferred_cat = sample_df.select_dtypes(include=["object","category","bool"]).columns.tolist()
            # only set if not already present
            if not numeric_cols:
                numeric_cols = inferred_num
            if not cat_cols:
                cat_cols = inferred_cat
        except Exception:
            pass

# final cast to lists
numeric_cols = list(numeric_cols)
cat_cols = list(cat_cols)

# Info for developer
st.sidebar.markdown("### Inference schema (best-effort)")
st.sidebar.write(f"Number of training columns: {len(train_columns)}")
st.sidebar.write("Numeric cols (inferred):")
st.sidebar.write(numeric_cols if numeric_cols else "EMPTY")
st.sidebar.write("Categorical cols (inferred):")
st.sidebar.write(cat_cols if cat_cols else "EMPTY")

# Page UI
st.title("Churn Prediction — Stable NN")
st.markdown(
    """
    **Input options**
    - Paste a JSON object with feature names matching training columns (preferred)
    - Or click "Load sample (first row)" to inspect training sample
    """
)

# Button to load first training row (if CSV exists in candidates)
if st.button("Load sample (first row)"):
    csv_path = find_file_in_candidates(TRAIN_CSV_NAME)
    if csv_path is None:
        st.warning(f"Training CSV not found ({TRAIN_CSV_NAME}) in repo root or /mnt/data")
    else:
        try:
            sample_df = pd.read_csv(csv_path)
            drop_cols = [c for c in sample_df.columns if c.lower() in ("churn","target","label","exited")]
            sample_df = sample_df.drop(columns=drop_cols, errors='ignore')
            st.dataframe(sample_df.head(1))
        except Exception as ex:
            st.error(f"Failed to load sample CSV: {ex}")

# Helper to prepare dataframe (align, normalize, coerce)
def prepare_input_df(raw_df: pd.DataFrame):
    df = raw_df.copy()

    # Ensure all train columns exist (add missing as NaN)
    for c in train_columns:
        if c not in df.columns:
            df[c] = np.nan

    # Keep only in training order
    df = df[train_columns].copy()

    # Normalize categorical columns: strip and title-case (stable)
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: safe_title_cat(v))

    # Coerce numeric columns
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Additional id-like columns: convert to string and strip
    id_like = [c for c in df.columns if c.lower().startswith("id") or c.lower().endswith("id")]
    for c in id_like:
        df[c] = df[c].astype(str).str.strip()

    return df

# Prediction helper
def predict_from_df(df: pd.DataFrame):
    # prepare
    df_prepped = prepare_input_df(df)

    # transform using preprocessor
    try:
        X_proc = preprocessor.transform(df_prepped)
    except Exception as ex:
        st.error(f"Preprocessor.transform failed: {ex}")
        if DEBUG:
            st.write("INPUT DF (attempt):")
            st.write(df_prepped.head(1))
            st.write("TRACEBACK:")
            st.text(traceback.format_exc())
        raise

    # Show debug transformed vector if requested
    if DEBUG:
        # show first row and first 50 features (if dense)
        try:
            rowvec = X_proc[0]
            # If sparse, convert to dense slice safely
            if hasattr(rowvec, "toarray"):
                row_dense = rowvec.toarray().ravel()
            else:
                row_dense = np.asarray(rowvec).ravel()
            st.write("TRANSFORMED VECTOR (first 80 values):")
            st.write(row_dense[:80].tolist())
        except Exception as ex:
            st.write("Could not display transformed vector:", ex)

    # run model predict_proba if available; else predict (sigmoid output)
    try:
        proba = float(model.predict(X_proc).ravel()[0])
    except Exception as ex:
        st.error(f"Model prediction failed: {ex}")
        st.stop()

    # stable thresholding and borderline detection
    threshold = 0.5
    # if the probability is extremely close to threshold, flag borderline
    borderline_margin = 1e-4
    borderline = abs(proba - threshold) < borderline_margin

    pred = int(proba >= threshold)
    label = None
    if label_encoder is not None:
        try:
            label = label_encoder.inverse_transform([pred])[0]
        except Exception:
            label = pred
    else:
        label = pred

    return proba, pred, label, borderline

# Option A — JSON
st.subheader("Option A — Paste JSON for a single row")
input_json = st.text_area("Feature JSON (single row)", value='{}', height=180)
if st.button("Predict from JSON"):
    try:
        if not input_json.strip():
            st.warning("Please paste a JSON object for a single row.")
        else:
            # tolerant parsing (accept Python dict notation too)
            try:
                parsed = safe_parse_json(input_json)
                if isinstance(parsed, dict):
                    df = pd.DataFrame([parsed])
                else:
                    st.error("JSON must represent a single object (dict).")
                    df = None
            except Exception as pe:
                st.error(f"JSON parse error: {pe}")
                df = None

            if df is not None:
                proba, pred, label, borderline = predict_from_df(df)
                st.metric("Predicted probability of churn", f"{proba:.6f}")
                if borderline:
                    st.warning("Borderline probability (very close to 0.5); result may be unstable.")
                st.write("Predicted label:", label)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        if DEBUG:
            st.text(traceback.format_exc())

# Option B — CSV row
st.subheader("Option B — Paste CSV (single row) or comma-separated features")
csv_text = st.text_area("Or paste a CSV row (header optional)", value="", height=120)
if st.button("Predict from CSV row"):
    try:
        if csv_text.strip() == "":
            st.warning("Paste a CSV row or header+row.")
        else:
            from io import StringIO
            df_try = pd.read_csv(StringIO(csv_text))
            if df_try.shape[0] > 1:
                df_try = df_try.head(1)
            proba, pred, label, borderline = predict_from_df(df_try)
            st.metric("Predicted probability of churn", f"{proba:.6f}")
            if borderline:
                st.warning("Borderline probability (very close to 0.5); result may be unstable.")
            st.write("Predicted label:", label)
    except Exception as e:
        st.error(f"CSV prediction failed: {e}")
        if DEBUG:
            st.text(traceback.format_exc())

st.markdown("---")
st.write("App expects artifacts saved in the repo root (or /mnt/data as fallback).")
st.write("To run locally: `pip install -r requirements.txt` then `streamlit run app.py`.")


# # app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import joblib
# import tensorflow as tf
# import json

# # Candidate paths (repo root first, then fallback)
# APP_ROOT = Path(__file__).parent.resolve()
# FALLBACK_DIR = Path("/mnt/data")

# CANDIDATE_DIRS = [APP_ROOT, FALLBACK_DIR]

# PREPROCESSOR_NAME = "preprocessor.joblib"
# TRAIN_COLUMNS_NAME = "train_columns.joblib"
# BEST_MODEL_NAME = "best_model.h5"
# FINAL_MODEL_NAME = "final_model.h5"
# LABEL_ENCODER_NAME = "label_encoder.joblib"
# TRAIN_CSV_NAME = "customer_churn_dataset-training-master.csv"  # optional sample

# def find_file_in_candidates(fname):
#     for d in CANDIDATE_DIRS:
#         p = d / fname
#         if p.exists():
#             return p
#     return None

# # Sidebar snapshot for debugging
# st.sidebar.markdown("### Files in candidate folders")
# for d in CANDIDATE_DIRS:
#     try:
#         entries = sorted([p.name for p in d.iterdir()]) if d.exists() else []
#         st.sidebar.markdown(f"**{d}** ({len(entries)})")
#         for e in entries[:30]:
#             st.sidebar.text(e)
#         if len(entries) > 30:
#             st.sidebar.text("... (truncated)")
#     except Exception as ex:
#         st.sidebar.text(f"cannot list {d}: {ex}")

# @st.cache_resource
# def load_artifacts():
#     # find preprocessor & train_columns
#     pre_path = find_file_in_candidates(PREPROCESSOR_NAME)
#     cols_path = find_file_in_candidates(TRAIN_COLUMNS_NAME)
#     if pre_path is None or cols_path is None:
#         raise RuntimeError(f"Missing artifacts. Expected {PREPROCESSOR_NAME} and {TRAIN_COLUMNS_NAME} in repo root or /mnt/data.")

#     pre = joblib.load(pre_path)
#     train_cols = joblib.load(cols_path)

#     # find model (prefer best_model then final_model)
#     best_path = find_file_in_candidates(BEST_MODEL_NAME)
#     final_path = find_file_in_candidates(FINAL_MODEL_NAME)
#     model_path = best_path if best_path is not None else final_path
#     model = None
#     if model_path is not None:
#         model = tf.keras.models.load_model(str(model_path))
#     else:
#         st.warning("No Keras model found (best_model.h5 or final_model.h5) in repo root or /mnt/data.")

#     # optional label encoder
#     le_path = find_file_in_candidates(LABEL_ENCODER_NAME)
#     le = None
#     if le_path is not None:
#         try:
#             le = joblib.load(le_path)
#         except Exception:
#             le = None

#     return pre, train_cols, model, le

# # Load artifacts
# try:
#     preprocessor, train_columns, model, label_encoder = load_artifacts()
# except Exception as e:
#     st.error(f"Failed to load artifacts: {e}")
#     st.stop()

# st.title("Churn Prediction — Neural Network")

# st.markdown(
#     """
#     **Input options**
#     - Paste a JSON object with feature names matching training columns (preferred)
#     - Or click "Load sample (first row)" to inspect training sample
#     """
# )

# # Button to load first training row (if CSV exists in candidates)
# if st.button("Load sample (first row)"):
#     # try candidate locations for the CSV
#     csv_path = find_file_in_candidates(TRAIN_CSV_NAME)
#     if csv_path is None:
#         st.warning(f"Training CSV not found ({TRAIN_CSV_NAME}) in repo root or /mnt/data")
#     else:
#         try:
#             sample_df = pd.read_csv(csv_path)
#             # drop common target-like columns if present
#             drop_cols = [c for c in sample_df.columns if c.lower() in ("churn","target","label")]
#             sample_df = sample_df.drop(columns=drop_cols, errors='ignore')
#             st.dataframe(sample_df.head(1))
#         except Exception as ex:
#             st.error(f"Failed to load sample CSV: {ex}")

# st.subheader("Option A — Paste JSON for a single row")
# input_json = st.text_area("Feature JSON (single row)", value='{}', height=180)
# if st.button("Predict from JSON"):
#     try:
#         row = pd.read_json(input_json, typ='series')
#         df = pd.DataFrame([row])
#         # Ensure columns match training columns: add missing cols with NaN
#         for c in train_columns:
#             if c not in df.columns:
#                 df[c] = np.nan
#         df = df[train_columns]
#         X_proc = preprocessor.transform(df)
#         proba = float(model.predict(X_proc).ravel()[0])
#         pred = int(proba >= 0.5)
#         label = label_encoder.inverse_transform([pred])[0] if label_encoder is not None else pred
#         st.metric("Predicted probability of churn", f"{proba:.4f}")
#         st.write("Predicted label:", label)
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")

# st.subheader("Option B — Paste CSV (single row) or comma-separated features")
# csv_text = st.text_area("Or paste a CSV row (header optional)", value="", height=120)
# if st.button("Predict from CSV row"):
#     try:
#         if csv_text.strip() == "":
#             st.warning("Paste a CSV row or disable this option.")
#         else:
#             from io import StringIO
#             df_try = pd.read_csv(StringIO(csv_text))
#             if df_try.shape[0] > 1:
#                 df_try = df_try.head(1)
#             for c in train_columns:
#                 if c not in df_try.columns:
#                     df_try[c] = np.nan
#             df_try = df_try[train_columns]
#             X_proc = preprocessor.transform(df_try)
#             proba = float(model.predict(X_proc).ravel()[0])
#             pred = int(proba >= 0.5)
#             label = label_encoder.inverse_transform([pred])[0] if label_encoder is not None else pred
#             st.metric("Predicted probability of churn", f"{proba:.4f}")
#             st.write("Predicted label:", label)
#     except Exception as e:
#         st.error(f"CSV prediction failed: {e}")

# st.markdown("---")
# st.write("App expects artifacts saved in the repo root (or /mnt/data as fallback).")
# st.write("To run locally: `pip install -r requirements.txt` then `streamlit run app.py`.")

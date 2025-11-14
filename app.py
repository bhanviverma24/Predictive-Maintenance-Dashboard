# app.py
"""
Reactive Streamlit dashboard for Predictive Maintenance.
Works with: MLP, 1D-CNN, Autoencoder.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import datetime

# ------------------------------------------------------
# Page Setup
# ------------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("Predictive Maintenance for IoT Machines (All Features)")

MODEL_DIR = "models"
DATA_PATH = "equipment_anomaly_data.csv"

# ------------------------------------------------------
# Load Dataset
# ------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset '{DATA_PATH}' not found.")
        st.stop()
    df = pd.read_csv(DATA_PATH)

    required = ['temperature', 'pressure', 'vibration', 'humidity',
                'equipment', 'location', 'faulty']
    for col in required:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()
    return df[required].dropna().reset_index(drop=True)

df = load_data()

st.subheader("Dataset preview (first 8 rows)")
st.dataframe(df.head(8))

# ------------------------------------------------------
# Load Models + Preprocessor + Metrics
# ------------------------------------------------------
@st.cache_resource
def load_resources():
    preproc = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    metrics_df = pd.read_csv(os.path.join(MODEL_DIR, "model_metrics.csv"))

    models = {}
    available = {
        "MLP": "mlp_model.keras",
        "CNN": "cnn_model.keras",
        "AUTOENCODER": "autoencoder_model.keras"
    }

    for name, file in available.items():
        path = os.path.join(MODEL_DIR, file)
        if os.path.exists(path):
            models[name] = load_model(path, compile=False)

    # autoencoder threshold
    th_path = os.path.join(MODEL_DIR, "autoencoder_threshold.npy")
    threshold = np.load(th_path) if os.path.exists(th_path) else None

    return preproc, models, metrics_df, threshold

preprocessor, models, metrics_df, threshold = load_resources()

st.subheader("Saved Model Performance")
st.dataframe(metrics_df)

# ------------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------------
st.sidebar.header("Manual Input")

temp = st.sidebar.slider("Temperature (°C)",
                         float(df.temperature.min()), float(df.temperature.max()), float(df.temperature.mean()))
pressure = st.sidebar.slider("Pressure (bar)",
                             float(df.pressure.min()), float(df.pressure.max()), float(df.pressure.mean()))
vibration = st.sidebar.slider("Vibration (normalized)",
                              float(df.vibration.min()), float(df.vibration.max()), float(df.vibration.mean()))
humidity = st.sidebar.slider("Humidity (%)",
                             float(df.humidity.min()), float(df.humidity.max()), float(df.humidity.mean()))
equipment = st.sidebar.selectbox("Equipment", sorted(df.equipment.unique()))
location = st.sidebar.selectbox("Location", sorted(df.location.unique()))

model_choice = st.sidebar.selectbox("Choose Model", sorted(models.keys()))

# ------------------------------------------------------
# Prepare Input
# ------------------------------------------------------
input_df = pd.DataFrame({
    "temperature": [temp],
    "pressure": [pressure],
    "vibration": [vibration],
    "humidity": [humidity],
    "equipment": [equipment],
    "location": [location]
})

X_input = preprocessor.transform(input_df)

# ------------------------------------------------------
# Prediction Logic
# ------------------------------------------------------
def predict(model_key, X):
    model = models[model_key]

    if model_key == "AUTOENCODER":
        recon = model.predict(X, verbose=0)
        mse = np.mean((X - recon) ** 2)
        is_fault = mse > threshold
        text = f"MSE = {mse:.6f} | Threshold = {threshold:.6f}"
        return mse, is_fault, text

    if model_key == "CNN":
        X_cnn = X.reshape(1, X.shape[1], 1)
        prob = model.predict(X_cnn, verbose=0).ravel()[0]
    else:  # MLP
        prob = model.predict(X, verbose=0).ravel()[0]

    is_fault = prob > 0.5
    return prob, is_fault, f"Probability = {prob:.4f}"

pred_value, is_fault, info_text = predict(model_choice, X_input)

# ------------------------------------------------------
# Output Display
# ------------------------------------------------------
st.subheader("Live Prediction")

st.write(info_text)

if model_choice == "AUTOENCODER":
    if is_fault:
        st.error("Prediction: FAULTY (Anomaly Detected)")
    else:
        st.success("Prediction: NORMAL")
else:
    if is_fault:
        st.error(f"FAULTY (prob={pred_value:.3f})")
    else:
        st.success(f"NORMAL (prob={pred_value:.3f})")

# ------------------------------------------------------
# Maintenance Schedule Suggestion
# ------------------------------------------------------
def maintenance_schedule(value, model_key):
    if model_key == "AUTOENCODER":
        sev = (value - threshold) / (threshold + 1e-9)
    else:
        sev = (value - 0.5) / 0.5

    if sev <= 0:
        return "No maintenance required", "Low", None
    if sev > 1.0:
        return "Maintenance Required", "High", datetime.datetime.now() + datetime.timedelta(hours=2)
    if sev > 0.5:
        return "Maintenance Suggested", "Medium", datetime.datetime.now() + datetime.timedelta(hours=8)
    return "Maintenance Suggested", "Low", datetime.datetime.now() + datetime.timedelta(hours=24)

status, priority, sug_time = maintenance_schedule(pred_value, model_choice)

st.info(f"Status: {status} | Priority: {priority}")

if sug_time:
    st.info(f"Suggested Maintenance Time: **{sug_time.strftime('%Y-%m-%d %H:%M')}**")

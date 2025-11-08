import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap, os, tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import datetime

# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("Predictive Maintenance for IoT Machines")
st.markdown("""
Use pre-trained **Deep Learning models (MLP, 1D-CNN, LSTM, Autoencoder)**  
to detect machine anomalies from **temperature** and **vibration** data.
""")

# ----------------------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    file_path = "equipment_anomaly_data.csv"
    if not os.path.exists(file_path):
        st.error(" Dataset file 'equipment_anomaly_data.csv' not found.")
        st.stop()
    df = pd.read_csv(file_path)
    df = df[['temperature', 'vibration', 'faulty']].dropna()
    return df

df = load_data()
st.subheader(" Dataset Preview")
st.dataframe(df.head(10))

# ----------------------------------------------------------------------
# Load Pre-trained Models and Scaler
# ----------------------------------------------------------------------
@st.cache_resource
def load_pretrained_models():
    try:
        custom_objects = {
            "mean_squared_error": tf.keras.losses.MeanSquaredError(),
            "accuracy": tf.keras.metrics.BinaryAccuracy()
        }

        scaler = joblib.load("models/scaler.pkl")
        mlp = load_model("models/mlp_model.h5", compile=False)
        cnn = load_model("models/cnn_model.h5", compile=False)
        lstm = load_model("models/lstm_model.h5", compile=False)
        autoencoder = load_model("models/autoencoder_model.h5",
                                 custom_objects=custom_objects, compile=False)
        threshold = np.load("models/autoencoder_threshold.npy")
    except Exception as e:
        st.error(f" Error loading models: {e}")
        st.stop()
    
    models_dict = {
        "MLP": mlp,
        "1D-CNN": cnn,
        "LSTM": lstm,
        "Autoencoder": autoencoder
    }
    return scaler, models_dict, threshold

scaler, models_dict, threshold = load_pretrained_models()

# ----------------------------------------------------------------------
# Display Model Performance Metrics
# ----------------------------------------------------------------------
st.subheader(" Model Performance Metrics")
metrics_df = pd.read_csv("models/model_metrics.csv")
st.dataframe(metrics_df)

# ----------------------------------------------------------------------
# SHAP Feature Importance (for MLP)
# ----------------------------------------------------------------------
@st.cache_resource
def compute_shap(model, sample_data):
    explainer = shap.Explainer(model, sample_data)
    shap_values = explainer(sample_data[:100])
    return shap_values

st.subheader(" Feature Importance (SHAP for MLP)")
sample_scaled = scaler.transform(df[['temperature', 'vibration']].values)
with st.spinner("Computing SHAP values..."):
    shap_values = compute_shap(models_dict["MLP"], sample_scaled)
    fig, ax = plt.subplots(figsize=(6, 4))
    shap.summary_plot(shap_values, sample_scaled[:100],
                      feature_names=['temperature', 'vibration'], show=False)
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------
# Predictive Maintenance Scheduling Function
# ----------------------------------------------------------------------
def schedule_maintenance(pred_value, model_type, threshold):
    """
    Returns maintenance status, priority, and suggested time
    """
    if model_type == "Autoencoder":
        is_fault = pred_value > threshold
        severity = (pred_value - threshold) / threshold
    else:
        is_fault = pred_value > 0.5
        severity = (pred_value - 0.5) / 0.5

    if not is_fault:
        return "No Maintenance Needed", "Low", None

    # Priority determination
    if severity >= 1.0:
        priority = "High"
        delta_hours = 2
    elif severity >= 0.5:
        priority = "Medium"
        delta_hours = 8
    else:
        priority = "Low"
        delta_hours = 24

    suggested_time = datetime.datetime.now() + datetime.timedelta(hours=delta_hours)
    suggested_time_str = suggested_time.strftime("%Y-%m-%d %H:%M")
    return "Maintenance Required", priority, suggested_time_str

# ----------------------------------------------------------------------
# Real-Time Prediction (All Models)
# ----------------------------------------------------------------------
st.subheader(" Real-Time Machine Fault Prediction")

model_choice = st.selectbox("Select Model", ["MLP", "1D-CNN", "LSTM", "Autoencoder"])

temp = st.slider("Temperature (°C)",
                 float(df['temperature'].min()), 
                 float(df['temperature'].max()), 
                 float(df['temperature'].mean()))

vib = st.slider("Vibration (m/s²)",
                float(df['vibration'].min()), 
                float(df['vibration'].max()), 
                float(df['vibration'].mean()))

sample_scaled = scaler.transform([[temp, vib]])

# ---------------------
# Prediction Section
# ---------------------
if model_choice == "MLP":
    pred = models_dict["MLP"].predict(sample_scaled, verbose=0)[0][0]
elif model_choice == "1D-CNN":
    pred = models_dict["1D-CNN"].predict(sample_scaled.reshape(1, 1, 2), verbose=0)[0][0]
elif model_choice == "LSTM":
    seq_input = np.tile(sample_scaled, (5, 1)).reshape(1, 5, 2)
    pred = models_dict["LSTM"].predict(seq_input, verbose=0)[0][0]
else:  # Autoencoder
    recon = models_dict["Autoencoder"].predict(sample_scaled, verbose=0)[0]
    pred = np.mean(np.square(sample_scaled - recon))

# ---------------------
# Maintenance Scheduling
# ---------------------
status, priority, suggested_time = schedule_maintenance(pred, model_choice, threshold)

if status == "No Maintenance Needed":
    st.success(f"Machine Status: HEALTHY | {status}")
else:
    st.error(f"Machine Status: FAULTY | {status}")
    st.info(f"Priority: {priority}")
    st.info(f"Suggested Maintenance Time: {suggested_time}")

# Display probability or MSE
if model_choice == "Autoencoder":
    st.caption(f"MSE: {pred:.4f} | Threshold: {threshold:.4f}")
else:
    st.caption(f"Fault Probability: {pred:.2f} | Threshold: 0.5")

st.caption("Developed by **Bhanvi Verma** | Pre-Trained Predictive Maintenance Dashboard")

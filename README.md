# Predictive Maintenance Dashboard

This project is an AI-powered **Predictive Maintenance Dashboard** that monitors industrial machines in real time and predicts potential faults before they occur. It uses deep learning models and an interactive dashboard to help industries reduce downtime, lower maintenance costs, and optimize machine performance.

---

## Features

* **Real-time machine fault prediction** using MLP and 1D-CNN models
* **Unsupervised anomaly detection** using Autoencoders
* **Automated maintenance scheduling** based on risk scores
* **Interactive dashboard** for monitoring machine health
* **Comprehensive log system** to store predictions, alerts, and maintenance tasks
* **Modular and scalable architecture** suitable for any industrial setup
* **Supports tabular sensor data** (temperature, pressure, vibration, humidity, etc.)

---

## System Overview

### Machine Learning Models

The system uses three deep learning models:

* **MLP (Multi-Layer Perceptron)** – Captures nonlinear behavior
* **1D-CNN** – Learns patterns in sensor value sequences
* **Autoencoder** – Detects anomalies without labeled data

Each model outputs:

* Fault probability
* Severity level
* Suggested maintenance action

### **Sensors & Input Data**

Supports multiple real-time inputs:

* Temperature
* Pressure
* Vibration
* Humidity
* Machine type
* Location

---

## ** Dashboard Features**

The Streamlit dashboard provides:

* Live visualization of machine health
* Fault probability and anomaly score graphs
* Maintenance task generator
* Historical logs and trends
* Model performance metrics
* Easy model switching (MLP, CNN, Autoencoder)

---

## ** Tech Stack**

| Component       | Technology          |
| --------------- | ------------------- |
| Programming     | Python              |
| ML Framework    | TensorFlow / Keras  |
| Data Handling   | Pandas, NumPy       |
| Visualization   | Matplotlib, Seaborn |
| Dashboard       | Streamlit           |
| Version Control | Git, GitHub         |

---

## ** Dataset**

This project uses an industrial equipment dataset containing:

* Temperature
* Pressure
* Vibration
* Humidity
* Machine Type
* Fault Label

Dataset Source: Kaggle — *Industrial Equipment Monitoring Dataset (DNKumars)*

---

## ** How to Run**

### **1. Clone the repository**

```bash
git clone https://github.com/bhanviverma24/Predictive-Maintenance-Dashboard.git
cd Predictive-Maintenance-Dashboard
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the dashboard**

```bash
streamlit run dashboard.py
```

---

## ** Folder Structure**

```
Predictive-Maintenance-Dashboard/
│── models/                 # Trained ML models
│── data/                   # Dataset files
│── logs/                   # Prediction + maintenance logs
│── dashboard.py            # Streamlit dashboard
│── preprocess.py           # Data preprocessing
│── train_models.py         # Training scripts (MLP, CNN, Autoencoder)
│── utils.py                # Helper functions
│── requirements.txt
│── README.md
```

---

## ** Future Improvements**

* Integration with IoT hardware for live sensor streaming
* Mobile dashboard view
* Model retraining on new data (self-learning system)
* Alert notifications via SMS/Email

---



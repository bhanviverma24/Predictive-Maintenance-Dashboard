# train_models.py

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATA_PATH = "equipment_anomaly_data.csv"
MODEL_DIR = "models"
EPOCHS = 20
BATCH_SIZE = 32
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------
df = pd.read_csv(DATA_PATH)

required_cols = [
    'temperature', 'pressure', 'vibration', 'humidity',
    'equipment', 'location', 'faulty'
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[required_cols].dropna().reset_index(drop=True)

num_cols = ['temperature', 'pressure', 'vibration', 'humidity']
cat_cols = ['equipment', 'location']
target = "faulty"

X = df[num_cols + cat_cols]
y = df[target].astype(int).values

# -------------------------------------------------------
# Preprocessing
# -------------------------------------------------------
preprocessor = ColumnTransformer([
    ("scale", StandardScaler(), num_cols),
    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols)
])

X_trans = preprocessor.fit_transform(X)

# Save preprocessor
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))
print("[OK] Saved preprocessor.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_trans, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# -------------------------------------------------------
# Metric Calculation (ONLY 3 METRICS)
# -------------------------------------------------------
def compute_metrics(y_true, y_pred_prob):
    y_pred = (np.array(y_pred_prob) > 0.5).astype(int)
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0)
    }

results = []

# -------------------------------------------------------
# 1) MLP (Recommended primary model)
# -------------------------------------------------------
mlp = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

mlp.compile(optimizer="adam", loss="binary_crossentropy")
mlp.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

mlp_preds = mlp.predict(X_test).ravel()
results.append(["MLP", *compute_metrics(y_test, mlp_preds).values()])

mlp.save(os.path.join(MODEL_DIR, "mlp_model.keras"))
print("[OK] Saved MLP model")

# -------------------------------------------------------
# 2) 1D-CNN (Tabular CNN)
# -------------------------------------------------------
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

cnn = models.Sequential([
    layers.Input(shape=(X_train.shape[1], 1)),
    layers.Conv1D(64, kernel_size=3, activation="relu"),
    layers.Conv1D(32, kernel_size=3, activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

cnn.compile(optimizer="adam", loss="binary_crossentropy")
cnn.fit(X_train_cnn, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

cnn_preds = cnn.predict(X_test_cnn).ravel()
results.append(["1D-CNN", *compute_metrics(y_test, cnn_preds).values()])

cnn.save(os.path.join(MODEL_DIR, "cnn_model.keras"))
print("[OK] Saved CNN model")

# -------------------------------------------------------
# 3) Autoencoder (Unsupervised anomaly detection)
# -------------------------------------------------------
X_train_norm = X_train[y_train == 0]  # only normal samples
if len(X_train_norm) < 20:
    X_train_norm = X_train  # fallback

auto = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(X_train.shape[1], activation="linear")
])

auto.compile(optimizer="adam", loss="mse")
auto.fit(X_train_norm, X_train_norm, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

recon = auto.predict(X_test)
mse = np.mean((X_test - recon) ** 2, axis=1)

recon_norm = auto.predict(X_train_norm)
mse_norm = np.mean((X_train_norm - recon_norm) ** 2, axis=1)
threshold = np.percentile(mse_norm, 95)

np.save(os.path.join(MODEL_DIR, "autoencoder_threshold.npy"), threshold)
auto.save(os.path.join(MODEL_DIR, "autoencoder_model.keras"))

y_pred_auto = (mse > threshold).astype(int)
results.append(["Autoencoder", *compute_metrics(y_test, y_pred_auto).values()])

# -------------------------------------------------------
# Save all metrics (ONLY 3 METRICS)
# -------------------------------------------------------
results_df = pd.DataFrame(results, columns=[
    "Model", "Precision", "Recall", "F1-score"
])
results_df.to_csv(os.path.join(MODEL_DIR, "model_metrics.csv"), index=False)

print("\nTraining Completed Successfully.")
print(results_df)

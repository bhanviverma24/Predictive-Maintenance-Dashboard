import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib
import os

# -----------------------------
#  Load and preprocess data
# -----------------------------
df = pd.read_csv("equipment_anomaly_data.csv")
X = df[['temperature', 'vibration']].values
y = df['faulty'].astype(int).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Create folder for saving
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# Results storage
results = []

# -----------------------------
# Helper function to compute metrics
# -----------------------------
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0)
    }

# -----------------------------
#  MLP Model
# -----------------------------
mlp = models.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

y_pred = (mlp.predict(X_test) > 0.5).astype(int)
metrics = compute_metrics(y_test, y_pred)
results.append(["MLP", *metrics.values()])

mlp.save("models/mlp_model.keras")
mlp.save("models/mlp_model.h5")
print("MLP trained and saved.")

# -----------------------------
#  1D-CNN Model
# -----------------------------
X_train_cnn = X_train.reshape((X_train.shape[0], 1, 2))
X_test_cnn = X_test.reshape((X_test.shape[0], 1, 2))

cnn = models.Sequential([
    layers.Input(shape=(1, 2)),
    layers.Conv1D(32, kernel_size=1, activation='relu'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_cnn, y_train, epochs=10, batch_size=32, verbose=0)

y_pred = (cnn.predict(X_test_cnn) > 0.5).astype(int)
metrics = compute_metrics(y_test, y_pred)
results.append(["1D-CNN", *metrics.values()])

cnn.save("models/cnn_model.keras")
cnn.save("models/cnn_model.h5")
print("1D-CNN trained and saved.")

# -----------------------------
#  LSTM Model
# -----------------------------
seq_len = 5
X_seq, y_seq = [], []
for i in range(len(X_scaled) - seq_len):
    X_seq.append(X_scaled[i:i + seq_len])
    y_seq.append(y[i + seq_len])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

lstm = models.Sequential([
    layers.Input(shape=(seq_len, 2)),
    layers.LSTM(64),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=0)

y_pred = (lstm.predict(X_test_seq) > 0.5).astype(int)
metrics = compute_metrics(y_test_seq, y_pred)
results.append(["LSTM", *metrics.values()])

lstm.save("models/lstm_model.keras")
lstm.save("models/lstm_model.h5")
print("LSTM trained and saved.")

# -----------------------------
#  Autoencoder Model
# -----------------------------
X_normal = X_train[y_train == 0]
autoencoder = models.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='linear')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_normal, X_normal, epochs=10, batch_size=32, verbose=0)

autoencoder.save("models/autoencoder_model.keras")
autoencoder.save("models/autoencoder_model.h5")

recon = autoencoder.predict(X_test)
mse = np.mean(np.square(X_test - recon), axis=1)
threshold = np.percentile(mse, 95)
np.save("models/autoencoder_threshold.npy", threshold)

auto_pred = (mse > threshold).astype(int)
metrics = compute_metrics(y_test, auto_pred)
results.append(["Autoencoder", *metrics.values()])

print(f"Autoencoder trained and saved. Threshold: {threshold:.6f}")

# -----------------------------
#  Save Metrics
# -----------------------------
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "ROC-AUC", "Precision", "Recall", "F1-score"])
results_df.to_csv("models/model_metrics.csv", index=False)

print("\n================= MODEL PERFORMANCE SUMMARY =================")
print(results_df.to_string(index=False))
print("==============================================================")
print("All models trained and saved successfully in 'models/' directory.")

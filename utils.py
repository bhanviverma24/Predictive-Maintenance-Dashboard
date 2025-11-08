import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    # Adjust target column as per dataset
    for col in ['faulty', 'failure', 'anomaly', 'label']:
        if col in df.columns:
            target_col = col
            break
    else:
        raise ValueError("No valid target column found.")

    features = [c for c in df.columns if c not in [target_col]]
    df = df[[c for c in features if c in ['temperature', 'vibration']] + [target_col]].dropna()

    X = df[['temperature', 'vibration']].values
    y = df[target_col].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# src/train_dummy_model.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib

os.makedirs("./models", exist_ok=True)
os.makedirs("./data", exist_ok=True)

motor_types = [
    'ID Fan Motor',
    'Dust Collector Fan Motor',
    'Automatic Mould-Handling Motor',
    'Hybrid Moulding Machine Motor',
    'Hydraulic Gesa Moulding Motor'
]

# create synthetic dataset with simple rule-based label
rows = []
np.random.seed(42)
for i in range(2000):
    m = np.random.choice(motor_types)
    # different distributions per motor type -- make dataset realistic-ish
    if 'ID Fan' in m:
        temp = np.random.normal(55, 6)
        vib = np.random.normal(2.5, 1.2)
        current = np.random.normal(12, 4)
    elif 'Dust Collector' in m:
        temp = np.random.normal(65, 8)
        vib = np.random.normal(6, 2)
        current = np.random.normal(18, 6)
    elif 'Automatic Mould' in m:
        temp = np.random.normal(75, 10)
        vib = np.random.normal(5, 2)
        current = np.random.normal(30, 8)
    elif 'Hybrid' in m:
        temp = np.random.normal(70, 9)
        vib = np.random.normal(6.5, 2.5)
        current = np.random.normal(40, 12)
    else:  # Hydraulic Gesa
        temp = np.random.normal(80, 12)
        vib = np.random.normal(7.5, 3)
        current = np.random.normal(55, 15)

    speed = np.random.normal(1500, 300)
    # rule-based label: maintenance needed if vibration high or temp high or current very high
    label = int((vib > 8) or (temp > 85) or (current > 60))
    rows.append((m, float(temp), float(vib), float(current), float(speed), label))

df = pd.DataFrame(rows, columns=['motor_type','temperature','vibration','current','speed','maintenance'])
df.to_csv("./data/sample_data.csv", index=False)

# pipeline
categorical = ['motor_type']
numeric = ['temperature','vibration','current','speed']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical)],
    remainder='passthrough'
)

model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=150, random_state=42))

X = df[['motor_type','temperature','vibration','current','speed']]
y = df['maintenance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

print("Train score:", model.score(X_train, y_train))
print("Test score:", model.score(X_test, y_test))

joblib.dump(model, "./models/motor_pipeline.pkl")
print("Saved pipeline to ./models/motor_pipeline.pkl and sample CSV to ../data/sample_data.csv")

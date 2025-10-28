import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

plt.figure(figsize=(12, 10))

df = pd.read_csv("./src/fitness_dataset.csv")

label_encoder = LabelEncoder()

# Tratamento de dados nulos e categóricos
df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce").fillna(df["sleep_hours"].median())

df["smokes"] = (
    df["smokes"].astype(str).str.strip().str.lower()
      .map({"yes": 1, "no": 0, "1": 1, "0": 0})
).astype(int)

df["gender"] = df["gender"].replace({"F": 0, "M": 1}).astype(int)

# BMI = peso(kg) / (altura(m))^2
h_m = pd.to_numeric(df["height_cm"], errors="coerce") / 100.0
bmi = pd.to_numeric(df["weight_kg"], errors="coerce") / (h_m**2)
df["bmi"] = bmi.replace([np.inf, -np.inf], np.nan).fillna(bmi.median())

# Features (com BMI, substitui altura e peso)
x = df[['age', 'bmi', 'heart_rate', 'blood_pressure',
        'sleep_hours', 'nutrition_quality', 'activity_index', 'smokes', 'gender']]
y = df['is_fit']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

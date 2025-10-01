import matplotlib.pyplot as plt
import pandas as pd

from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

plt.figure(figsize=(12, 10))

df = pd.read_csv("./src/fitness_dataset.csv")

label_encoder = LabelEncoder()

#Tratamento de dados nulos e categóricos
df = pd.read_csv("./src/fitness_dataset.csv")

df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())

df["smokes"] = (
    df["smokes"].astype(str).str.strip().str.lower()
      .map({"yes": 1, "no": 0, "1": 1, "0": 0})
).astype(int)

df["gender"] = df["gender"].replace({"F": 0, "M": 1}).astype(int)

# Carregar o conjunto de dados
x = df[['age', 'height_cm', 'weight_kg', 'heart_rate', 'blood_pressure',
       'sleep_hours', 'nutrition_quality', 'activity_index', 'smokes',
       'gender']]
y = df['is_fit']

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
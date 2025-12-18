import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

# 1. Carregando a base
df = pd.read_csv("./src/fitness_dataset.csv")

# 2. Pré-processamento (igual aos outros exercícios)

# Copia para evitar mexer direto no df original
df = df.copy()

# 2.1. Tratamento de valores ausentes
df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())

# 2.2. Padronização de 'smokes' (no/yes/0/1 → 0/1)
df["smokes"] = df["smokes"].astype(str).str.strip().str.lower()
map_smokes = {"no": 0, "0": 0, "yes": 1, "1": 1}
df["smokes"] = df["smokes"].map(map_smokes).astype(int)

# 2.3. Codificação de 'gender' (F/M → 0/1)
df["gender"] = df["gender"].map({"F": 0, "M": 1}).astype(int)

# 2.4. Criação do BMI
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

# 3. Definição de features e alvo
features = [
    "age",
    "heart_rate",
    "blood_pressure",
    "sleep_hours",
    "nutrition_quality",
    "activity_index",
    "smokes",
    "gender",
    "bmi",
]

X = df[features]
y = df["is_fit"]

# 4. Divisão treino / teste (70/30 com estratificação)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# 5. Pipeline SVM: padronização + SVC RBF
svm_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
    ]
)

# 6. Treinamento
svm_pipeline.fit(X_train, y_train)

# 7. Predições
y_pred = svm_pipeline.predict(X_test)

# 8. Métricas
acc = accuracy_score(y_test, y_pred)
bacc = balanced_accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)

print("=== Support Vector Machine (SVM) - Kernel RBF ===\n")
print(f"Acurácia (accuracy): {acc:.3f}")
print(f"Acurácia balanceada (balanced accuracy): {bacc:.3f}\n")

print("Matriz de confusão:")
print(cm, "\n")

print("Relatório de classificação:")
print(report)

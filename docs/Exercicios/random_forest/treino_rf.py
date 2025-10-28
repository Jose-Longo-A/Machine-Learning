import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
)

# ---------- carregar/prepare X,y exatamente como na "divisaodados_rf.py" ----------
df = pd.read_csv("./src/fitness_dataset.csv")

df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce").fillna(df["sleep_hours"].median())
df["smokes"] = (
    df["smokes"].astype(str).str.strip().str.lower()
      .map({"yes": 1, "no": 0, "1": 1, "0": 0})
).astype(int)
df["gender"] = df["gender"].replace({"F": 0, "M": 1}).astype(int)

h_m = pd.to_numeric(df["height_cm"], errors="coerce") / 100.0
bmi = pd.to_numeric(df["weight_kg"], errors="coerce") / (h_m**2)
df["bmi"] = bmi.replace([float("inf"), float("-inf")], pd.NA).fillna(bmi.median())

num_cols = ["age","heart_rate","blood_pressure","sleep_hours","nutrition_quality","activity_index","bmi"]
cat_cols = ["smokes","gender"]
target = "is_fit"

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

X = df[num_cols + cat_cols].values
y = df[target].astype(int).values
# -------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest baseline (ajuste livre depois)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

acc  = accuracy_score(y_test, pred)
bacc = balanced_accuracy_score(y_test, pred)
print(f"Acurácia: {acc:.4f}")
print(f"Balanced Accuracy: {bacc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, pred, digits=4))

# Matriz de confusão (SVG)
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Não Fit (pred)","Fit (pred)"],
            yticklabels=["Não Fit (real)","Fit (real)"])
plt.title("Matriz de Confusão — Random Forest")
plt.xlabel("Previsto"); plt.ylabel("Real")
plt.tight_layout()

buf = StringIO()
plt.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
print(buf.getvalue())
plt.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# === Carregar e tratar como na base_tratada ===
df = pd.read_csv("./src/fitness_dataset.csv")

# sleep_hours -> mediana
df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())

# smokes -> normaliza rótulos e converte p/ 0/1
df["smokes"] = (
    df["smokes"].astype(str).str.strip().str.lower()
      .map({"yes": 1, "no": 0, "1": 1, "0": 0})
).astype(int)

# gender -> F=0, M=1
df["gender"] = df["gender"].replace({"F": 0, "M": 1}).astype(int)

# bmi = weight_kg / (height_m^2)
h_m = pd.to_numeric(df["height_cm"], errors="coerce") / 100.0
bmi = pd.to_numeric(df["weight_kg"], errors="coerce") / (h_m ** 2)
df["bmi"] = (
    bmi.replace([float("inf"), float("-inf")], pd.NA)
       .fillna(bmi.median())
)

# === Features e alvo ===
# Usa BMI e remove height_cm/weight_kg para evitar redundância
num_cols = [
    "age", "heart_rate", "blood_pressure", "sleep_hours",
    "nutrition_quality", "activity_index", "bmi"
]
cat_cols = ["smokes", "gender"]
target = "is_fit"

# Garantir numéricos válidos nas contínuas
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

X_num = df[num_cols]
X_cat = df[cat_cols]

# Padronização (KNN é sensível à escala)
scaler = StandardScaler()
X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=num_cols)

# Matriz final
X = pd.concat([X_num_scaled.reset_index(drop=True),
               X_cat.reset_index(drop=True)], axis=1).values
y = df[target].astype(int).values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# PCA 2D só para visualização/decisão do gráfico
pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_2d, y_train)
pred = knn.predict(X_test_2d)

# Métricas
acc = accuracy_score(y_test, pred)
bacc = balanced_accuracy_score(y_test, pred)
print(f"Acurácia: {acc:.4f}")

# Plot
plt.figure(figsize=(12, 10))
h = 0.05
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(
    x=X_train_2d[:, 0], y=X_train_2d[:, 1], hue=y_train,
    palette="deep", s=100, edgecolor="k", alpha=0.8, legend=True
)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KNN Decision Boundary (Fitness Dataset, com BMI)")
plt.tight_layout()

# Renderiza 1 SVG no site e fecha a figura
buf = StringIO()
plt.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
print(buf.getvalue())
plt.close()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Só pra ter certeza que o chunk rodou
print("Gerando fronteira de decisão do SVM...\n")

# 1. Carregando a base
df = pd.read_csv("./src/fitness_dataset.csv").copy()

# 2. Mesmo pré-processamento de sempre

# Valores ausentes em sleep_hours
df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())

# smokes: normalização e mapeamento para 0/1
df["smokes"] = df["smokes"].astype(str).str.strip().str.lower()
map_smokes = {"no": 0, "0": 0, "yes": 1, "1": 1}
df["smokes"] = df["smokes"].map(map_smokes).astype(int)

# gender: F/M -> 0/1
df["gender"] = df["gender"].map({"F": 0, "M": 1}).astype(int)

# BMI
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

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

# 3. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.30,
  random_state=42,
  stratify=y,
)

# 4. Padronização + PCA para 2D (apenas para visualização)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)

# 5. Treina SVM em 2D
svm_2d = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_2d.fit(X_train_pca, y_train)

# 6. Cria malha para fronteira de decisão
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
  np.linspace(x_min, x_max, 300),
  np.linspace(y_min, y_max, 300),
)

Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 7. Plot (no padrão dos outros gráficos: SVG via StringIO)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5])

# pontos de treino
plt.scatter(
  X_train_pca[y_train == 0, 0],
  X_train_pca[y_train == 0, 1],
  label="Não fit (treino)",
  alpha=0.7,
  marker="o",
)

plt.scatter(
  X_train_pca[y_train == 1, 0],
  X_train_pca[y_train == 1, 1],
  label="Fit (treino)",
  alpha=0.7,
  marker="^",
)

plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.title("Fronteira de decisão do SVM (PCA 2D)")
plt.legend()
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()
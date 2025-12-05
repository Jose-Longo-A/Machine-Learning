import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Carregar base ---
df = pd.read_csv("./src/fitness_dataset.csv")

# --- Pré-processamento ---
df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce").fillna(df["sleep_hours"].median())

df["smokes"] = (
    df["smokes"].astype(str).str.strip().str.lower()
      .map({"yes": 1, "no": 0, "1": 1, "0": 0})
).astype(int)

df["gender"] = df["gender"].replace({"F": 0, "M": 1}).astype(int)

# BMI
h_m = pd.to_numeric(df["height_cm"], errors="coerce") / 100.0
bmi = pd.to_numeric(df["weight_kg"], errors="coerce") / (h_m**2)
df["bmi"] = bmi.replace([float("inf"), float("-inf")], pd.NA).fillna(bmi.median())

# Features (X) e alvo (y apenas para avaliação futura)
num_cols = ["age", "heart_rate", "blood_pressure", "sleep_hours",
            "nutrition_quality", "activity_index", "bmi"]
cat_cols = ["smokes", "gender"]
target = "is_fit"

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

X_num = df[num_cols]
X_cat = df[cat_cols]

scaler = StandardScaler()
X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=num_cols)

X = pd.concat([X_num_scaled.reset_index(drop=True),
               X_cat.reset_index(drop=True)], axis=1).values
y = df[target].astype(int).values

# --- K-Means ---
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300,
                random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# --- PCA p/ visualização ---
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

# --- Plot ---
plt.figure(figsize=(12, 10))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
# projeta centróides para o espaço PCA
centroids_2d = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
            c='red', marker='*', s=200, label='Centroids (PCA proj.)')

plt.title('K-Means Clustering (Fitness Dataset)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()

import base64
from io import BytesIO
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # backend headless p/ rodar no mkdocs
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1) Carregar base
df = pd.read_csv("./src/wine.csv")

# 2) Remover campos verbosos/irrelevantes (se existirem)
drop_cols = ["Unnamed: 0", "title", "description", "taster_twitter_handle"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# 3) Conversões e imputações mínimas
df["points"] = pd.to_numeric(df.get("points"), errors="coerce")
if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].fillna(df["price"].median())

# 4) AMOSTRAGEM cedo (para não pesar no build)
MAX_ROWS = 5000
if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=27).reset_index(drop=True)

# 5) Seleção de features (evitar cardinalidade extrema)
num_cols = [c for c in ["price"] if c in df.columns]
cat_cols = [c for c in ["country", "province", "region_1", "variety"] if c in df.columns]  # sem taster_name

for c in cat_cols:
    df[c] = df[c].fillna("Unknown").astype(str).str.strip()

X_cat = pd.get_dummies(df[cat_cols], drop_first=False, dtype=int) if cat_cols else pd.DataFrame(index=df.index)
X_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)

if not X_num.empty:
    scaler = StandardScaler()
    X_num[num_cols] = scaler.fit_transform(X_num[num_cols])

X = pd.concat([X_num, X_cat], axis=1).values

# 6) PCA 2D para visualização
pca = PCA(n_components=2, random_state=27)
X_pca = pca.fit_transform(X)

# 7) K-Means no espaço 2D (k=3)
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=27)
labels = kmeans.fit_predict(X_pca)

# 8) Plot (matplotlib puro) e saída em base64 (HTML <img>)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=18, alpha=0.85)
plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    c="red", marker="*", s=200, label="Centróides"
)
plt.title("K-Means (Wine) — PCA 2D, k=3")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(loc="best")

buf = BytesIO()
plt.savefig(buf, format="png", transparent=True, bbox_inches="tight")
plt.close()
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode("utf-8")
print(f'<img src="data:image/png;base64,{img_b64}" alt="KMeans clustering (Wine)"/>')

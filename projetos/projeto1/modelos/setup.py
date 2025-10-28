import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # backend headless para rodar no mkdocs
import matplotlib.pyplot as plt

from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1) Carregar base
df = pd.read_csv("./src/wine.csv")

# 2) Limpeza básica e alvo
drop_cols = ["Unnamed: 0", "title", "description", "taster_twitter_handle"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

df["points"] = pd.to_numeric(df.get("points"), errors="coerce")
if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

df = df[df["points"].notna()].copy()
df["quality_high"] = (df["points"] >= 90).astype(int)

# 3) Amostragem estratificada cedo (antes de dummies) para visualização
MAX_VIS = 4000
if len(df) > MAX_VIS:
    frac = MAX_VIS / len(df)
    df = (
        df.groupby("quality_high", group_keys=False)
          .apply(lambda g: g.sample(frac=frac, random_state=42))
          .reset_index(drop=True)
    )

y = df["quality_high"].values

# 4) Features (evitar cardinalidade absurda)
num_cols = [c for c in ["price"] if c in df.columns]
cat_cols = [c for c in ["country", "province", "region_1", "variety"] if c in df.columns]

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna("Unknown").astype(str).str.strip()

X_cat = pd.get_dummies(df[cat_cols], drop_first=False, dtype=int) if cat_cols else pd.DataFrame(index=df.index)
X_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)

if not X_num.empty:
    scaler = StandardScaler()
    X_num[num_cols] = scaler.fit_transform(X_num[num_cols])

X = pd.concat([X_num, X_cat], axis=1).values

# 5) Split (70/30), manter consistência
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=27, stratify=y
)

# 6) PCA (2D) apenas para visualização/fronteira
pca = PCA(n_components=2, random_state=27)
X_train_2d = pca.fit_transform(X_train)
X_test_2d  = pca.transform(X_test)

# 7) KNN no espaço 2D projetado
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_2d, y_train)
pred = knn.predict(X_test_2d)
acc = accuracy_score(y_test, pred)
print(f"Acurácia (KNN sklearn c/ PCA 2D, k=5): {acc:.3f}")

# 8) Fronteira de decisão em 2D (grid menos denso pra não travar)
plt.figure(figsize=(8, 6))
h = 0.25

x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, s=16, edgecolors="k", alpha=0.8)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KNN — Fronteira de decisão (PCA 2D)")
plt.tight_layout()

buf = StringIO()
plt.savefig(buf, format="svg", transparent=True)
print(buf.getvalue())
plt.close()

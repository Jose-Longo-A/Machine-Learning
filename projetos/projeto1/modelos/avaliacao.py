import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1) Carregar base
df = pd.read_csv("./src/wine.csv")

# 2) Limpeza e alvo derivado
drop_cols = ["Unnamed: 0", "title", "description", "taster_twitter_handle"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

df["points"] = pd.to_numeric(df.get("points"), errors="coerce")
df = df[df["points"].notna()].copy()  # necessário para quality_high

if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].fillna(df["price"].median())

df["quality_high"] = (df["points"] >= 90).astype(int)
y = df["quality_high"].values

# 3) AMOSTRAGEM estratificada cedo (para não pesar no build)
MAX_ROWS = 6000
if len(df) > MAX_ROWS:
    frac = MAX_ROWS / len(df)
    df = (
        df.groupby("quality_high", group_keys=False)
          .apply(lambda g: g.sample(frac=frac, random_state=27))
          .reset_index(drop=True)
    )
y = df["quality_high"].values

# 4) Features (evitar cardinalidade extrema)
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

# 5) Split (70/30) + PCA no treino
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=27, stratify=y
)

pca = PCA(n_components=2, random_state=27)
X_train_pca = pca.fit_transform(X_train)

# 6) K-Means em treino (k=3) e mapeamento cluster->classe por voto majoritário
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=27)
labels_train = kmeans.fit_predict(X_train_pca)

cluster_map = {}
classes = np.unique(y_train)
for c in np.unique(labels_train):
    mask = labels_train == c
    if mask.sum() == 0:
        cluster_map[c] = classes[0]
        continue
    counts = np.bincount(y_train[mask], minlength=classes.max() + 1)
    cluster_map[c] = counts.argmax()

# 7) Acurácia e matriz de confusão (TREINO), como no seu exemplo
y_pred_train = np.array([cluster_map[c] for c in labels_train])
acc = accuracy_score(y_train, y_pred_train)
cm = confusion_matrix(y_train, y_pred_train, labels=np.sort(classes))

cm_df = pd.DataFrame(
    cm,
    index=[f"Classe Real {cls}" for cls in np.sort(classes)],
    columns=[f"Classe Pred {cls}" for cls in np.sort(classes)]
)

print(f"Acurácia (mapeamento por voto, treino): {acc*100:.2f}%")
print("<br>Matriz de Confusão (treino):")
print(cm_df.to_html(index=True))

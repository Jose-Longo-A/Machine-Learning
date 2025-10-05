import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# =================== Carregar base ===================
df = pd.read_csv("./src/wine.csv")

# Remover colunas verbosas/irrelevantes, se existirem
drop_cols = ["Unnamed: 0", "title", "description", "taster_twitter_handle"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Converter 'points' e 'price' para numérico
df["points"] = pd.to_numeric(df.get("points"), errors="coerce")
if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Manter linhas com points (necessário para o alvo)
df = df[df["points"].notna()].copy()
df["quality_high"] = (df["points"] >= 90).astype(int)

# ===== Amostragem estratificada cedo (antes de dummies) p/ não pesar o build =====
MAX_ROWS = 5000
if len(df) > MAX_ROWS:
    frac = MAX_ROWS / len(df)
    df = (
        df.groupby("quality_high", group_keys=False)
          .apply(lambda g: g.sample(frac=frac, random_state=42))
          .reset_index(drop=True)
    )

# ===== Seleção de features =====
# Evitar cardinalidade muito alta (sem taster_name/designation/winery/region_2)
num_cols = [c for c in ["price"] if c in df.columns]
cat_cols = [c for c in ["country", "province", "region_1", "variety"] if c in df.columns]

# Imputações simples
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna("Unknown").astype(str).str.strip()

# One-hot para categóricas
X_cat = pd.get_dummies(df[cat_cols], drop_first=False, dtype=int) if cat_cols else pd.DataFrame(index=df.index)
X_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)

# Escala nas numéricas
if not X_num.empty:
    scaler = StandardScaler()
    X_num[num_cols] = scaler.fit_transform(X_num[num_cols])

# Concat final
X = pd.concat([X_num, X_cat], axis=1).values
y = df["quality_high"].values

# Split (70/30) com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=27, stratify=y
)

# ===== Implementação manual do KNN =====
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        for x in X:
            dists = np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
            k_idx = np.argpartition(dists, self.k)[:self.k]
            k_labels = self.y_train[k_idx]
            vals, counts = np.unique(k_labels, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds)

knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia (KNN manual, k={knn.k}): {acc:.3f}")
print(f"Tamanho treino/teste: {X_train.shape[0]} / {X_test.shape[0]}")
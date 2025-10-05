import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ===== 1) Ler base =====
df = pd.read_csv("./src/wine.csv")

# ===== 2) Remover colunas irrelevantes (se existirem) =====
drop_cols = ["Unnamed: 0", "title", "description", "taster_twitter_handle"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# ===== 3) Conversões de tipo essenciais =====
if "points" in df.columns:
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

# ===== 4) Criar alvo supervisionado a partir de 'points' =====
# quality_high = 1 (>= 90), 0 (< 90)
# Linhas sem 'points' não ajudam na classificação — removemos
df = df[df["points"].notna()].copy()
df["quality_high"] = (df["points"] >= 90).astype(int)

# ===== 5) Codificar variáveis categóricas (LabelEncoder simples) =====
cat_cols = [
    col for col in ["country", "province", "region_1", "region_2",
                    "taster_name", "designation", "variety", "winery"]
    if col in df.columns
]

for c in cat_cols:
    le = LabelEncoder()
    df[c] = df[c].fillna("Unknown").astype(str)
    df[c] = le.fit_transform(df[c])

# ===== 6) Definir X (features) e y (alvo), evitando vazamento de 'points' =====
feature_cols = []
feature_cols += [c for c in cat_cols]              # categóricas codificadas
if "price" in df.columns:
    feature_cols.append("price")                   # numérica útil
# NÃO usar 'points' como feature (origem do alvo)
X = df[feature_cols].copy()
y = df["quality_high"].copy()

# ===== 7) Divisão treino/teste com estratificação =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=27,
    stratify=y
)
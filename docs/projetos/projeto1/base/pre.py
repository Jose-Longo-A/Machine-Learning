import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ===== 1) Ler base =====
df = pd.read_csv("./src/wine.csv")

# ===== 2) Remover colunas irrelevantes (se existirem) =====
drop_cols = ["Unnamed: 0", "title", "description", "taster_twitter_handle"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# ===== 3) Garantir tipos numéricos e tratar nulos =====
if "points" in df.columns:
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

for c in num_cols:
    med = df[c].median() if df[c].notna().any() else 0
    df[c] = df[c].fillna(med)

for c in cat_cols:
    df[c] = df[c].fillna("Unknown")

# ===== 4) Criar alvo supervisionado (se houver points) =====
if "points" in df.columns:
    df["quality_high"] = (df["points"] >= 90).astype(int)

# ===== 5) Codificação simples de categóricas (rastreável) =====
df_ready = df.copy()
for c in df_ready.select_dtypes(exclude=["number"]).columns:
    le = LabelEncoder()
    df_ready[c] = le.fit_transform(df_ready[c].astype(str))

# ===== 6) Evitar vazamento: retirar 'points' das features "base" =====
if "points" in df_ready.columns:
    df_ready = df_ready.drop(columns=["points"])

# ===== 7) Mostrar SOMENTE algumas linhas para não poluir o markdown =====
n_show = min(20, len(df_ready))
print(f"Dimensão (após pré-processamento): {df_ready.shape[0]} linhas x {df_ready.shape[1]} colunas\n")
print(df_ready.sample(n=n_show, random_state=42).to_markdown(index=False))

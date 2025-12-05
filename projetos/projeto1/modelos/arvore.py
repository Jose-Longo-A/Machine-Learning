import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ===================== Ler base =====================
df = pd.read_csv("./src/wine.csv")

# ===================== Excluir colunas não desejadas =====================
# (mesma ideia do seu script antigo: tirar campos irrelevantes/verbosos)
drop_cols = ["Unnamed: 0", "title", "description", "taster_twitter_handle"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# ===================== Conversões básicas =====================
# 'points' será usada para criar o alvo; por isso NÃO entra em X
df["points"] = pd.to_numeric(df["points"], errors="coerce")

# price pode ter nulos; converte e imputa mediana
if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if df["price"].isna().any():
        df["price"] = df["price"].fillna(df["price"].median())

# manter somente linhas com 'points' (necessário para o alvo)
df = df[df["points"].notna()].copy()

# ===================== Criar alvo supervisionado =====================
# quality_high = 1 (>= 90), 0 (< 90)
df["quality_high"] = (df["points"] >= 90).astype(int)

# ===================== Label encoding das categóricas =====================
# (seguindo o espírito do seu antigo: LabelEncoder direto nas colunas de texto)
cat_cols = [c for c in ["country", "province", "region_1", "region_2",
                        "taster_name", "designation", "variety", "winery"]
            if c in df.columns]

for c in cat_cols:
    le = LabelEncoder()
    df[c] = df[c].fillna("Unknown").astype(str)
    df[c] = le.fit_transform(df[c])

# ===================== Definição de features (X) e alvo (y) =====================
# Evitar vazamento: NUNCA usar 'points' em X, pois dela deriva o alvo
feature_cols = []
feature_cols += cat_cols
if "price" in df.columns:
    feature_cols.append("price")

x = df[feature_cols].copy()
y = df["quality_high"].copy()

# ===================== Divisão treino/teste =====================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=27, stratify=y
)

# ===================== Criar e treinar o modelo de árvore de decisão =====================
classifier = tree.DecisionTreeClassifier(random_state=27)
classifier.fit(x_train, y_train)

# ===================== Avaliar o modelo =====================
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisão da Validação: {accuracy:.2f}")

# Importância das features
feature_importance = pd.DataFrame({
    'Feature': classifier.feature_names_in_,
    'Importância': classifier.feature_importances_
}).sort_values(by='Importância', ascending=False)

print("<br>Importância das Features:")
print(feature_importance.to_html(index=False))

# ===================== Plot da árvore (até profundidade 5, como no seu padrão) =====================
plt.figure(figsize=(20, 10))
try:
    tree.plot_tree(classifier, max_depth=5, fontsize=10, feature_names=classifier.feature_names_in_)
except Exception:
    tree.plot_tree(classifier, max_depth=5, fontsize=10)

# Para imprimir na página HTML (MkDocs)
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
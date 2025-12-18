import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================
# 1. Carregamento da base
# =========================
df = pd.read_csv("./src/ProjetoInter4_Dados_Unidos.csv").copy()

# =========================
# 2. Pré-processamento
# =========================
# datas
if "game_launch" in df.columns:
    df["game_launch"] = pd.to_datetime(df["game_launch"], errors="coerce")
    df["launch_year"] = df["game_launch"].dt.year
else:
    df["launch_year"] = np.nan

# mantém só linhas com target
df = df.dropna(subset=["game_user"]).copy()

# normaliza strings (evita categoria " Sony " != "sony")
for col in ["publisher_name", "genre_name"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    else:
        df[col] = "unknown"

# imputação de ano
df["launch_year"] = pd.to_numeric(df["launch_year"], errors="coerce")
df["launch_year"] = df["launch_year"].fillna(df["launch_year"].median())

# target encoding (média do target por categoria)
global_mean = df["game_user"].mean()
for col in ["publisher_name", "genre_name"]:
    means = df.groupby(col)["game_user"].mean()
    df[col] = df[col].map(means).fillna(global_mean)

# =========================
# 3. Features e alvo
# =========================
features = ["publisher_name", "genre_name", "launch_year"]
X = df[features].copy()
y = df["game_user"].copy()

# garantia final: sem NaN
X["publisher_name"] = X["publisher_name"].fillna(global_mean)
X["genre_name"] = X["genre_name"].fillna(global_mean)
X["launch_year"] = X["launch_year"].fillna(X["launch_year"].median())

# =========================
# 4. Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. Modelo
# =========================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# 6. Métricas
# =========================
print("=== Regressão Linear ===")
print(f"R²  : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")

# =========================
# 7. Gráfico coeficientes
# =========================
coef = model.named_steps["lr"].coef_
coef_df = pd.DataFrame({"Variável": features, "Coeficiente": coef}).sort_values("Coeficiente")

plt.figure(figsize=(8, 4))
plt.barh(coef_df["Variável"], coef_df["Coeficiente"])
plt.title("Regressão Linear – Coeficientes")
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()

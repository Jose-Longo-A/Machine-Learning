import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# =========================
# 1. Base
# =========================
df = pd.read_csv("./src/ProjetoInter4_Dados_Unidos.csv").copy()

# =========================
# 2. Pré-processamento
# =========================
if "game_launch" in df.columns:
    df["game_launch"] = pd.to_datetime(df["game_launch"], errors="coerce")
    df["launch_year"] = df["game_launch"].dt.year
else:
    df["launch_year"] = np.nan

df = df.dropna(subset=["game_user"]).copy()

for col in ["publisher_name", "genre_name"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    else:
        df[col] = "unknown"

df["launch_year"] = pd.to_numeric(df["launch_year"], errors="coerce")
df["launch_year"] = df["launch_year"].fillna(df["launch_year"].median())

global_mean = df["game_user"].mean()
for col in ["publisher_name", "genre_name"]:
    means = df.groupby(col)["game_user"].mean()
    df[col] = df[col].map(means).fillna(global_mean)

# =========================
# 3. Features / alvo
# =========================
features = ["publisher_name", "genre_name", "launch_year"]
X = df[features].copy()
y = df["game_user"].copy()

# garantia final
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
    ("knn", KNeighborsRegressor(n_neighbors=15, weights="distance"))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# 6. Métricas
# =========================
print("=== KNN Regressor ===")
print(f"R²  : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")

# =========================
# 7. Permutation Importance
# =========================
perm = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

idx = perm.importances_mean.argsort()

plt.figure(figsize=(8, 4))
plt.barh([features[i] for i in idx], perm.importances_mean[idx])
plt.title("KNN – Importância das Variáveis")
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()

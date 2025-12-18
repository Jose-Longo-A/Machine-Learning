import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================
# 1. Base
# =========================
df = pd.read_csv("./src/ProjetoInter4_Dados_Unidos.csv")

df["game_launch"] = pd.to_datetime(df["game_launch"], errors="coerce")
df["launch_year"] = df["game_launch"].dt.year
df = df.dropna(subset=["game_user"])

for col in ["publisher_name", "genre_name"]:
    if col in df.columns:
        means = df.groupby(col)["game_user"].mean()
        df[col] = df[col].map(means)

df["launch_year"] = df["launch_year"].fillna(df["launch_year"].median())

features = ["publisher_name", "genre_name", "launch_year"]
X = df[features]
y = df["game_user"]

# =========================
# 2. Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. Modelo
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# 4. Métricas
# =========================
print("=== Random Forest ===")
print(f"R²  : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")

# =========================
# 5. Importância
# =========================
imp = pd.DataFrame({
    "Variável": features,
    "Importância": model.feature_importances_
}).sort_values("Importância")

plt.figure(figsize=(8, 4))
plt.barh(imp["Variável"], imp["Importância"])
plt.title("Random Forest – Importância das Variáveis")
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()

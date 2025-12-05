import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar base
df = pd.read_csv("./src/fitness_dataset.csv")

# Tratamento
df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce").fillna(df["sleep_hours"].median())

df["smokes"] = (
    df["smokes"].astype(str).str.strip().str.lower()
      .map({"yes": 1, "no": 0, "1": 1, "0": 0})
).astype(int)

df["gender"] = df["gender"].replace({"F": 0, "M": 1}).astype(int)

# Criar BMI e substituir height/weight
h_m = pd.to_numeric(df["height_cm"], errors="coerce") / 100.0
bmi = pd.to_numeric(df["weight_kg"], errors="coerce") / (h_m**2)
df["bmi"] = bmi.replace([float("inf"), float("-inf")], pd.NA).fillna(bmi.median())

# Features e alvo
num_cols = ["age", "heart_rate", "blood_pressure", "sleep_hours",
            "nutrition_quality", "activity_index", "bmi"]
cat_cols = ["smokes", "gender"]
target = "is_fit"

# Garantir numéricos válidos
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

X_num = df[num_cols]
X_cat = df[cat_cols]

# Padronizar atributos numéricos
scaler = StandardScaler()
X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=num_cols)

# Concatenar
X = pd.concat([X_num_scaled.reset_index(drop=True),
               X_cat.reset_index(drop=True)], axis=1).values
y = df[target].astype(int).values

# Divisão treino/teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Tamanho treino:", X_train.shape[0])
print("Tamanho teste:", X_test.shape[0])
print("Proporção classes treino:", y_train.mean())
print("Proporção classes teste:", y_test.mean())

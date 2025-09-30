import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./src/fitness_dataset.csv")

# sleep_hours
df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce").fillna(df["sleep_hours"].median())

# smokes -> 0/1
df["smokes"] = (
    df["smokes"].astype(str).str.strip().str.lower()
      .map({"yes": 1, "no": 0, "1": 1, "0": 0})
).astype(int)

# gender -> 0/1
df["gender"] = df["gender"].replace({"F": 0, "M": 1}).astype(int)

# BMI
h_m = pd.to_numeric(df["height_cm"], errors="coerce") / 100.0
bmi = pd.to_numeric(df["weight_kg"], errors="coerce") / (h_m**2)
df["bmi"] = bmi.replace([float("inf"), float("-inf")], pd.NA).fillna(bmi.median())

# Features e alvo (y só para avaliação externa)
num_cols = ["age","heart_rate","blood_pressure","sleep_hours","nutrition_quality","activity_index","bmi"]
cat_cols = ["smokes","gender"]
target = "is_fit"

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

X_num = df[num_cols]
X_cat = df[cat_cols]

# padroniza só numéricos
scaler = StandardScaler()
X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=num_cols)

# matriz final p/ K-Means
X = pd.concat([X_num_scaled.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1).values
y = df[target].astype(int).values

print("Shape X:", X.shape)
print("Proporção 'is_fit'=1:", y.mean())
print("Observação: K-Means usa APENAS X; y é para avaliar depois.")

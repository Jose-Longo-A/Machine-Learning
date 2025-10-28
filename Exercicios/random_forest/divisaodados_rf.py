import pandas as pd

df = pd.read_csv("./src/fitness_dataset.csv")

# sleep_hours -> mediana
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

# Features e alvo
num_cols = ["age","heart_rate","blood_pressure","sleep_hours","nutrition_quality","activity_index","bmi"]
cat_cols = ["smokes","gender"]
target = "is_fit"

# garantir numéricos válidos
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

X = df[num_cols + cat_cols].values
y = df[target].astype(int).values

print("Shape X:", X.shape)
print("Proporção 'is_fit'=1:", y.mean())

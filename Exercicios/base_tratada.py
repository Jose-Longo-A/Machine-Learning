import pandas as pd

df = pd.read_csv("./src/fitness_dataset.csv")

df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())

df["smokes"] = (
    df["smokes"].astype(str).str.strip().str.lower()
      .map({"yes": 1, "no": 0, "1": 1, "0": 0})
).astype(int)

df["gender"] = df["gender"].replace({"F": 0, "M": 1}).astype(int)

h_m = pd.to_numeric(df["height_cm"], errors="coerce") / 100.0
bmi = pd.to_numeric(df["weight_kg"], errors="coerce") / (h_m**2)
df["bmi"] = bmi.replace([float("inf"), float("-inf")], pd.NA).fillna(bmi.median())

print(df.sample(n=15).to_markdown(index=False))
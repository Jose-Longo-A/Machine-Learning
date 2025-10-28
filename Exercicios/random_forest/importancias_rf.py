import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# mesmo preparo
df = pd.read_csv("./src/fitness_dataset.csv")
df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce").fillna(df["sleep_hours"].median())
df["smokes"] = (
    df["smokes"].astype(str).str.strip().str.lower()
      .map({"yes": 1, "no": 0, "1": 1, "0": 0})
).astype(int)
df["gender"] = df["gender"].replace({"F": 0, "M": 1}).astype(int)
h_m = pd.to_numeric(df["height_cm"], errors="coerce") / 100.0
bmi = pd.to_numeric(df["weight_kg"], errors="coerce") / (h_m**2)
df["bmi"] = bmi.replace([float("inf"), float("-inf")], pd.NA).fillna(bmi.median())

num_cols = ["age","heart_rate","blood_pressure","sleep_hours","nutrition_quality","activity_index","bmi"]
cat_cols = ["smokes","gender"]
target = "is_fit"

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

feat_cols = num_cols + cat_cols
X = df[feat_cols].values
y = df[target].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2, max_features="sqrt",
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
order = importances.argsort()
plt.figure(figsize=(8,6))
sns.barplot(x=importances[order], y=[feat_cols[i] for i in order], orient="h")
plt.title("Importância das Variáveis — Random Forest")
plt.xlabel("Importância"); plt.ylabel("Atributo")
plt.tight_layout()

buf = StringIO()
plt.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
print(buf.getvalue())
plt.close()

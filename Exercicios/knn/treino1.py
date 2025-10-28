import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./src/fitness_dataset.csv")

# Tratamento dos dados
df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())
df["smokes"] = df["smokes"].replace({"yes": 1, "no": 0, "1": 1, "0": 0}).astype(int)
df["gender"] = df["gender"].replace({"F": 0, "M": 1}).astype(int)

# bmi = weight_kg / (height_m^2)
h_m = pd.to_numeric(df["height_cm"], errors="coerce") / 100.0
bmi = pd.to_numeric(df["weight_kg"], errors="coerce") / (h_m ** 2)
df["bmi"] = (
    bmi.replace([float("inf"), float("-inf")], pd.NA)
       .fillna(bmi.median())
)


# Features e target
num_cols = [
    "age", "heart_rate", "blood_pressure",
    "sleep_hours", "nutrition_quality", "activity_index", "bmi"
]
cat_cols = ["smokes", "gender"]
target = "is_fit"

# Garantir tipos numéricos
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

X_num = df[num_cols]
X_cat = df[cat_cols]

# Padronização dos dados numéricos
scaler = StandardScaler()
X_num = pd.DataFrame(scaler.fit_transform(X_num), columns=num_cols)

# Junta tudo
X = pd.concat([X_num, X_cat.reset_index(drop=True)], axis=1).values
y = df[target].astype(int).values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
        k_idx = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_idx]
        vals, counts = np.unique(k_labels, return_counts=True)
        return vals[np.argmax(counts)]

# Treinar e avaliar
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia (KNN k={knn.k}): {acc:.2f}")
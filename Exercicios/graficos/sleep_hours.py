import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/fitness_dataset.csv")

# Conta nulos
nulos = df["sleep_hours"].isnull().sum()

# Conta as faixas numéricas (ignorando nulos)
count = pd.cut(df["sleep_hours"].dropna(), bins=7).value_counts().sort_index()

# Adiciona os nulos ao resultado (usando concat)
count = pd.concat([count, pd.Series({"Nulo": nulos})])

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(count.index.astype(str), count.values, color="#5A6C4F")
ax.set_title("Composição da coluna sleep_hours")
ax.set_ylabel("Quantidade de sono ")
plt.xticks(rotation=15)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
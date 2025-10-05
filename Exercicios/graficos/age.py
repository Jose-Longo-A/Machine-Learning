import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/fitness_dataset.csv")

# Conta as faixas de idade (ignorando nulos)
bins = [18, 25, 35, 45, 55, 65, 100]
labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
faixas = pd.cut(df["age"].dropna(), bins=bins, labels=labels, right=False)
count = faixas.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(count.index.astype(str), count.values, color="#5A6C4F")
ax.set_title("Distribuição das faixas de idade")
ax.set_ylabel("Quantidade")
plt.xticks(rotation=15)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
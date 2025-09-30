import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/fitness_dataset.csv")

# Conta os valores únicos da coluna is_fit (ignorando nulos)
fit = df["is_fit"].value_counts()

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(fit.index.astype(str), fit.values, color="#5A6C4F")
ax.set_title("Distribuição de pessoas fitness")
ax.set_xlabel("É fitness?")
ax.set_ylabel("Quantidade")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
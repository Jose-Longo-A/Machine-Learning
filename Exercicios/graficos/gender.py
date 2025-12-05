import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/fitness_dataset.csv")

# Conta os valores únicos da coluna gender (ignorando nulos)
generos = df["gender"].value_counts()

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(generos.index.astype(str), generos.values, color="#5A6C4F")
ax.set_title("Distribuição por gênero")
ax.set_xlabel("Gênero")
ax.set_ylabel("Quantidade")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
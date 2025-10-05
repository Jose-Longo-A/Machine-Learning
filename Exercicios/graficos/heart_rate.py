import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/fitness_dataset.csv")

# Remove nulos da coluna heart_rate
frequencia = df["heart_rate"].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(frequencia, bins=15, color="#5A6C4F", edgecolor="black")
ax.set_title("Distribuição da frequência cardíaca (bpm)")
ax.set_xlabel("Frequência cardíaca (bpm)")
ax.set_ylabel("Quantidade")
plt.grid(axis='y', alpha=0.3)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
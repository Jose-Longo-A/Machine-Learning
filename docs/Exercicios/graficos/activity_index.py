import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/fitness_dataset.csv")

# Remove nulos da coluna activity_index
atividade = df["activity_index"].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(atividade, bins=10, color="#5A6C4F", edgecolor="black")
ax.set_title("Distribuição do índice de atividade")
ax.set_xlabel("Índice de atividade")
ax.set_ylabel("Quantidade")
plt.grid(axis='y', alpha=0.3)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
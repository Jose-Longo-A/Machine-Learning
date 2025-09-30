import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/fitness_dataset.csv")

# Remove nulos da coluna height_cm
altura = df["height_cm"].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(altura, bins=15, color="#5A6C4F", edgecolor="black")
ax.set_title("Distribuição da altura (cm)")
ax.set_xlabel("Altura (cm)")
ax.set_ylabel("Quantidade")
plt.grid(axis='y', alpha=0.3)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
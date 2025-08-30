import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/fitness_dataset.csv")

# Remove nulos da coluna nutrition_quality
nutricao = df["nutrition_quality"].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(nutricao, bins=10, color="#5A6C4F", edgecolor="black")
ax.set_title("Distribuição da qualidade nutricional")
ax.set_xlabel("Qualidade nutricional")
ax.set_ylabel("Quantidade")
plt.grid(axis='y', alpha=0.3)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
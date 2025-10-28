import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/fitness_dataset.csv")

# Remove nulos da coluna weight_kg
peso = df["weight_kg"].dropna()

fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot(peso, vert=True, patch_artist=True, boxprops=dict(facecolor="#5A6C4F"))
ax.set_title("Boxplot do peso (kg)")
ax.set_ylabel("Peso (kg)")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
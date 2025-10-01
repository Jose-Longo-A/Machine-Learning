import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/fitness_dataset.csv")

# Remove nulos da coluna blood_pressure
pressao = df["blood_pressure"].dropna()

fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot(pressao, vert=True, patch_artist=True, boxprops=dict(facecolor="#5A6C4F"))
ax.set_title("Boxplot da pressão arterial")
ax.set_ylabel("Pressão arterial")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
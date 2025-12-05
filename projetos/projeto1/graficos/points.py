import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import numpy as np

df = pd.read_csv("./src/wine.csv")
pts = df["points"].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(pts, bins=20, edgecolor="black")
ax.set_title("Distribuição de pontuações (points)")
ax.set_xlabel("Pontos")
ax.set_ylabel("Frequência")

# Linha da média
if len(pts) > 0:
    m = float(np.mean(pts))
    ax.axvline(m, linestyle="--", linewidth=1.5)
    ax.text(m, ax.get_ylim()[1]*0.9, f"Média: {m:.1f}", rotation=90, va="top")

plt.tight_layout()
buf = StringIO(); plt.savefig(buf, format="svg", transparent=True); print(buf.getvalue())
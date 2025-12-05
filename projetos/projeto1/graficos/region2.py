import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/wine.csv")
counts = df["region_2"].dropna().value_counts().head(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(counts.index[::-1], counts.values[::-1])
ax.set_title("Sub-regiões secundárias (region_2) — Top 15")
ax.set_xlabel("Quantidade")
plt.tight_layout()

buf = StringIO(); plt.savefig(buf, format="svg", transparent=True); print(buf.getvalue())
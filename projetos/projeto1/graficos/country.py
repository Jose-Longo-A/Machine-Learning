# docs/projetos/projeto1/graficos/country.py
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/wine.csv")

counts = df["country"].dropna().value_counts()

top = counts.head(10)
outros = counts.iloc[10:].sum()

# pandas >= 2.0: usar concat em vez de Series.append
if outros > 0:
    plot_series = pd.concat([top, pd.Series({"Outros": outros})])
else:
    plot_series = top

labels = [str(x) for x in plot_series.index]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(labels, plot_series.values)
ax.set_title("Registros por país (Top 10 + Outros)")
ax.set_xlabel("País")
ax.set_ylabel("Quantidade")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

buf = StringIO()
plt.savefig(buf, format="svg", transparent=True)
print(buf.getvalue())
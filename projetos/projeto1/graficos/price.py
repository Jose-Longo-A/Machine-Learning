import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import numpy as np

df = pd.read_csv("./src/wine.csv")
price = pd.to_numeric(df["price"], errors="coerce").dropna()
if len(price) > 0:
    cap = np.nanpercentile(price, 99)  # corta 1% topo p/ reduzir assimetria
    price = price.clip(upper=cap)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(price, bins=30, edgecolor="black")
ax.set_title("Distribuição de preços (price) — com cap no 99º pct")
ax.set_xlabel("Preço (USD)")
ax.set_ylabel("Frequência")
plt.tight_layout()

buf = StringIO(); plt.savefig(buf, format="svg", transparent=True); print(buf.getvalue())

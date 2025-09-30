import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/fitness_dataset.csv")

count = pd.Series(df["smokes"]).value_counts()

fig, ax = plt.subplots(figsize=(8, 4))

ax.pie(count, labels=count.index, colors=["#5A6C4F", "#4F6C8C", "#C2A33D", "#A65A54"], autopct="%1.1f%%")

ax.set_title("Composição da coluna")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
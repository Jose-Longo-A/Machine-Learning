import pandas as pd

df = pd.read_csv("./src/fitness_dataset.csv")

df["sleep_hours"] = df["sleep_hours"].fillna("null")

# Exibe apenas 10 linhas aleatórias em formato markdown
print(df.sample(n=15).to_markdown(index=False))
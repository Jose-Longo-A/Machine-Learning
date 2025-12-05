import pandas as pd

df0 = pd.read_csv("./src/wine.csv")

# Ocultar campos muito verbosos na visualização “original”
hide_cols = {"description", "title", "taster_twitter_handle"}
show_cols = [c for c in df0.columns if c not in hide_cols]

df0_view = df0[show_cols].copy()

n_show = min(20, len(df0_view))
print(f"Dimensão (base original): {df0_view.shape[0]} linhas x {df0_view.shape[1]} colunas\n")
print(df0_view.sample(n=n_show, random_state=42).to_markdown(index=False))

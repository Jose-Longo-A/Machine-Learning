import pandas as pd

# Aqui assumimos que metricas_knn e metricas_kmeans
# já foram definidos pelos arquivos:
# - docs/Exercicios/metricas/knn_metricas.py
# - docs/Exercicios/metricas/kmeans_metricas.py

tabela = pd.DataFrame([metricas_knn, metricas_kmeans])

print("Comparação das métricas (KNN x K-Means):")
print(tabela.to_markdown(index=False))

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Aqui assumimos que y e labels já existem,
# pois foram definidos em "docs/Exercicios/k-means/treino.py"

cm = confusion_matrix(y, labels)

acc  = accuracy_score(y, labels)
prec = precision_score(y, labels)
rec  = recall_score(y, labels)
f1   = f1_score(y, labels)

print("Matriz de confusão - K-Means (clusters vs is_fit):")
print(cm)
print()
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

# Guarda em dicionário para comparação depois
metricas_kmeans = {
    "modelo": "K-Means (k=2)",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
}

# Gráfico da matriz de confusão
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=["Cluster 0", "Cluster 1"],
    yticklabels=["Real 0", "Real 1"]
)
plt.xlabel("Cluster atribuído")
plt.ylabel("Classe real (is_fit)")
plt.title("Matriz de Confusão - K-Means")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()

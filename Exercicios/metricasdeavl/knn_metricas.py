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

# Aqui assumimos que y_test e pred já existem,
# porque foram definidos em docs/Exercicios/knn/resultado1.py

cm = confusion_matrix(y_test, pred)

acc  = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec  = recall_score(y_test, pred)
f1   = f1_score(y_test, pred)

print("Matriz de confusão - KNN:")
print(cm)
print()
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

# Opcional: gráfico da matriz de confusão no mesmo estilo dos outros gráficos
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Real 0", "Real 1"])
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - KNN")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()

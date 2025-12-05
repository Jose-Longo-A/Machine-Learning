import matplotlib as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from io import StringIO
from sklearn import tree
from sklearn.metrics import accuracy_score

df = pd.read_csv("./src/fitness_dataset.csv")

print(df.head(5)) # Mostra as linhas do DataFrame
print(df.info())  # Mostra informações sobre o DataFrame, como tipos de dados e valores nulos


# VERIFICAÇÃO E CORREÇÃO DE DADOS NULOS NA BASE

print(df.isnull().values.any()) # verificar se há nulos
print(df.isnull().sum()) # mostrar a quantidade de nulos por coluna

# verifiquei que há nulos na coluna sleep_hours(160 nulos)

for col in df.columns:
    print(f"{col}:{(df[col] == 'NaN').sum()} ") # tentativa de contar nulos (deu errado)

print(df[df["sleep_hours"].isnull()]) # segunda tentativa de contar nulos (deu certo)

df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median()) # substitui os valores nulos pela mediana da coluna

print(df["sleep_hours"]) # printando a coluna sleep_hours para ver se os nulos foram substituídos
print(df[df["sleep_hours"].isnull()]) # última verificação de nulos na coluna sleep_hours


# TRATAMENTO DE VARIÁVEIS CATEGÓRICAS

df["gender"]
df["gender"].value_counts() # mostra a contagem de valores únicos

df["smokes"]
df["smokes"].value_counts() # mostra a contagem de valores únicos

# com essa contagem de valores únicos podemos ver que nas duas colunas há apenas 2 valores únicos, então podemos substituir esses valores por 0 e 1

df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median()) # substitui os valores nulos pela mediana da coluna
df["sleep_hours"]
df["smokes"] = df["smokes"].replace({"yes": 1, "no": 0}) # transforma os "yes" e "no" dos fumantes em 1 e 0
df["smokes"]


col = df.columns # coloca o nome das colunas em uma variável
print(col) # mostra o nome das colunas


print(df.describe())

# Tratamento de dados categóricos, transformando em numéricos


print(df["sleep_hours"].isnull().sum())  # deve dar 0
print(df.head())

df["gender"] = df["gender"].replace({"F": 0, "M": 1})

print(df.head())



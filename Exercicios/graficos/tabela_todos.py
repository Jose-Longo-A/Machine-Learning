import matplotlib as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from io import StringIO
from sklearn import tree
from sklearn.metrics import accuracy_score

df = pd.read_csv("./src/fitness_dataset.csv")

print(df.head(25))
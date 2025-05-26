import pandas as pd
import seaborn as sns

penguins = sns.load_dataset("penguins")

print(penguins.head())

print(penguins.isnull().sum())

df = penguins.dropna()
print(df.isnull().sum())
print("\nTotal missing values:", df.isnull().sum().sum())
print(df.info())

X = df.drop("species", axis=1)
y = df["species"]

X = pd.get_dummies(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Dataset split")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)
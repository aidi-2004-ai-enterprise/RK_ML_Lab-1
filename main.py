### Person A

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


### Person B

##XGBoost Model

#Import libraries
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib

#Encode target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

#Create and train XGBoost classifier (default and fixed parameters)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train_encoded)

#Predict on test set
y_pred = model.predict(X_test)

#Print Results
ConfusionMatrix = confusion_matrix(y_test_encoded, y_pred) 
ClassificationReport = classification_report(y_test_encoded, y_pred, target_names=le.classes_)
Accuracy = accuracy_score(y_test_encoded, y_pred) #Just for kicks

print("Estimator: XGBoost")
print(ConfusionMatrix)
print(ClassificationReport)
print(f"XGBoost Model Accuracy: {Accuracy:.4f}")

#Model Feature Importance
xgb.plot_importance(model)
plt.show()

#Save Model
joblib.dump(model, "penguin_xgboost_model.pkl")
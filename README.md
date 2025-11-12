# Wine-Quality-Prediction-Project-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("winequality-red.csv")
print("Dataset Shape:", data.shape)
print(data.head())

sns.heatmap(data.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.countplot(x="quality", data=data)
plt.title("Wine Quality Distribution")
plt.show()

X = data.drop("quality", axis=1)
y = data["quality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)
sgd_acc = accuracy_score(y_test, sgd_pred)

svc = SVC(random_state=42)
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
svc_acc = accuracy_score(y_test, svc_pred)

print("Random Forest Accuracy:", rf_acc)
print("SGD Classifier Accuracy:", sgd_acc)
print("SVC Accuracy:", svc_acc)

models = ["Random Forest", "SGD", "SVC"]
accuracies = [rf_acc, sgd_acc, svc_acc]
sns.barplot(x=models, y=accuracies)
plt.title("Model Accuracy Comparison")
plt.show()

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred))
print("Confusion Matrix (Random Forest):")
print(confusion_matrix(y_test, rf_pred))

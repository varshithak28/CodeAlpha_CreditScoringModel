import pandas as pd
data = pd.read_csv("credit_risk_dataset.csv")
print(data.head())
print(data.shape)
print(data.columns)
print(data.isnull().sum())
 
#Removing Missing values
data = data.dropna()
print("New Shape:", data.shape)

#Converting Text to Numbers form
data = pd.get_dummies(data, drop_first=True)
print(data.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Features and target
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression(max_iter=2000)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
 
#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

#Graph
plt.bar(["Accuracy"], [accuracy])
plt.ylabel("Score")
plt.title("Credit Scoring Model Accuracy")
plt.savefig("accuracy_graph.png")
print("Graph Saved")
# plt.show()
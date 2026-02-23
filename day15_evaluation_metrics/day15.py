import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Dataset
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([0,0,0,0,1,1,1,1])

# Train Model
model = LogisticRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Confusion Matrix
cm = confusion_matrix(y, predictions)

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n")
print(classification_report(y, predictions))

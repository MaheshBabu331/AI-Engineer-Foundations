from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np

X, y = load_iris(return_X_y=True)

# split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# train models
model1 = DecisionTreeClassifier()
model2 = SVC(probability=True)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# predictions (probabilities)
pred1 = model1.predict_proba(X_val)
pred2 = model2.predict_proba(X_val)

# average predictions
final_pred = (pred1 + pred2) / 2

# final prediction
y_pred = np.argmax(final_pred, axis=1)

# accuracy
accuracy = np.mean(y_pred == y_val)

print("Accuracy:", accuracy)

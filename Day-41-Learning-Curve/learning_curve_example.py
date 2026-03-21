from sklearn.datasets import load_iris
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X, y = load_iris(return_X_y=True)

model = DecisionTreeClassifier()

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5
)

print("Train Accuracy:", np.mean(train_scores, axis=1))
print("Test Accuracy:", np.mean(test_scores, axis=1))

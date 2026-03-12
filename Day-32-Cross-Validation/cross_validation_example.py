from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

data = load_iris()

X = data.data
y = data.target

model = SVC()

scores = cross_val_score(model, X, y, cv=5)

print("Scores:", scores)
print("Average Accuracy:", scores.mean())

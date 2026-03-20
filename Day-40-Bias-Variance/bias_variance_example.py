from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# load data
X, y = load_iris(return_X_y=True)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 🔴 Simple model → High Bias
model1 = DecisionTreeClassifier(max_depth=1)
model1.fit(X_train, y_train)

print("Simple Model")
print("Train:", model1.score(X_train, y_train))
print("Test:", model1.score(X_test, y_test))

# 🔴 Complex model → High Variance
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)

print("\nComplex Model")
print("Train:", model2.score(X_train, y_train))
print("Test:", model2.score(X_test, y_test))

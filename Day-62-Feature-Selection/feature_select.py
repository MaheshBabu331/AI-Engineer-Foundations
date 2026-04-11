import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression

X = np.array([
    [1, 100, 5],
    [2, 200, 6],
    [3, 300, 7],
    [4, 400, 8]
])

y = np.array([10, 20, 30, 40])

# select best 2 features
selector = SelectKBest(score_func=f_regression, k=2)
X_new = selector.fit_transform(X, y)

print(X_new)

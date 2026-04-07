from sklearn.linear_model import Ridge, Lasso
import numpy as np

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([1,2,3,4,5])

# Ridge (L2)
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print("Ridge prediction:", ridge.predict([[6]]))

# Lasso (L1)
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
print("Lasso prediction:", lasso.predict([[6]]))

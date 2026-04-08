from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import numpy as np

X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([20000, 28000, 35000, 45000, 52000, 60000])

model = Ridge()

# parameters to test
params = {'alpha': [0.1, 1, 10, 100]}

# grid search
grid = GridSearchCV(model, params, cv=3)
grid.fit(X, y)

print("Best alpha:", grid.best_params_)
print("Best score:", grid.best_score_)

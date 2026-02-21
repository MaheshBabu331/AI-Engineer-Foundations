import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

#  Hyperparameter Tuning (GridSearchCV)
x=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y=np.array([2,4,6,8,10,12,14,16,18,20])

#Base Model
model=Ridge()

#Hyperparameter Grid
param_grid={
  'alpha':[0.01,0.1,1,10,100]
}
#Grid search with 5-fold
grid=GridSearchCV(
  model,
  param_grid,
  cv=5,
  scoring="neg_mean_squared_error"
)
grid.fit(x,y)
print("Best Alpha:",grid.best_params_)
print("Best Score (MSE):", -grid.best_score_)

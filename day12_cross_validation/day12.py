import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
#Simple Data
x=np.array([
  [1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y=np.array([2,4,6,8,10,12,14,16,18,20])

#Model
model=LinearRegression()
#5-fold cross validation
scores=cross_val_score(
  model,
  x,
  y,
  cv=5,
  scoring="neg_mean_squared_error"
)
#Convert Negative MSE to Positive MSE
mse_scores= -scores
print("MSE For each fold:", mse_scores)
print("Average MSE:", mse_scores.mean())

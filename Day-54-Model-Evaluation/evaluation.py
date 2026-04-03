import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([20000, 28000, 35000, 45000, 52000, 60000])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("MAE:", int(mean_absolute_error(y, y_pred)))
print("MSE:", int(mean_squared_error(y, y_pred)))
print("R2:", round(r2_score(y, y_pred),2))

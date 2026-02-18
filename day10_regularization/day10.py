import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

#Regularization With Ridge

#Simple Dataset
x=np.array([[1],[2],[3],[4],[5]])
y=np.array([1,4,9,16,25])

# Create high-degree polynomial features (overfitting risk)
poly=PolynomialFeatures(degree=5)
x_poly=poly.fit_transform(x)


# Ridge with small alpha (weak regularization)
ridge_small=Ridge(alpha=0.1)
ridge_small.fit(x_poly,y)
pred_small=ridge_small.predict(x_poly)

# Ridge with large alpha (strong regularization)
ridge_large = Ridge(alpha=100)
ridge_large.fit(x_poly, y)
pred_large = ridge_large.predict(x_poly)

# Calculate errors
mse_small=mean_squared_error(y,pred_small)
mse_large=mean_squared_error(y,pred_large)

print("Small Alpha Coefficients \n:",ridge_small.coef_)
print("Large Alpha Coefficients \n:",ridge_large.coef_)
print("MSE Small:",mse_small)
print("MSE Large:",mse_large)



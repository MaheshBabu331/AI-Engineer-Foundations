import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Lasso Regression (Feature Selection)
#taking random data with three features
#Third feature is mostly irrelevant

x=np.array([
    [1, 5, 100],
    [2, 6, 200],
    [3, 7, 300],
    [4, 8, 400],
    [5, 9, 500]
])
# Targets Depends on First mainly two features
y=np.array([10, 14, 18, 22, 26])

#Lasso Model
lasso=Lasso(alpha=0.5)
lasso.fit(x,y)
predictions=lasso.predict(x)
mse=mean_squared_error(y,predictions)
print("Coefficient :",lasso.coef_)
print("Intercept :", lasso.intercept_)
print("MSE:",mse)

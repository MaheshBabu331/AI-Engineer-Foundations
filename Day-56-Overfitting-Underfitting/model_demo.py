import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# data (curve)
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([1,4,9,16,25])

# underfitting (simple model)
model1 = LinearRegression()
model1.fit(X, y)
print("Underfit prediction:", model1.predict([[6]]))

# better model
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model2 = LinearRegression()
model2.fit(X_poly, y)
print("Better prediction:", model2.predict(poly.transform([[6]]))
     )

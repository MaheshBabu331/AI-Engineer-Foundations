import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Polynomial Regression
#Equation is y=w1x^2+w2x+b 


# Feature (1D array)
x=np.array([[1],[2],[3],[4],[5]])

##Target (Quadratic pattern)
y=np.array([1,4,9,16,25])

#Create Polynomial Feature (Degree=2)
poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x)

#Train Model
model=LinearRegression()
model.fit(x_poly,y)

#predictions
predictions=model.predict(x_poly)

#Calculate Error
mse=mean_squared_error(y,predictions)

print("Transformed Features \n:",x_poly)
print("Predictions:",predictions)
print("Mean Squared Error:",mse)




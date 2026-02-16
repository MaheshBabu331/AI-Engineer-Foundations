import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Train/ Test Split  & Evaluation
#features (experience, Skill)
x=np.array([
  [1,5],
  [2,6],
  [3,7],
  [4,8],
  [5,9],
  [6,10],
  [7,11],
  [8,12]
])

#Target Salaries
y=np.array([23,26,28,30,33,38,41,45])

#Split Datasets 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=42)

#Create MOdel
model=LinearRegression()
#Train Model
model.fit(x_train,y_train)
#Predictions
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

#Calculate Errors:
train_mse=mean_squared_error(y_train,train_pred)
test_mse=mean_squared_error(y_test,test_pred)
print("Training MSE:", train_mse)
print("Testing MSE:",test_mse)

#Interpretation:

if test_mse > train_mse:
  print("Test Error Greater than slightly Train Error")








import numpy as np

#Features:
#Coloumn 0: Experience
#coloumn 1: Skills

x=np.array([
  [2,4],
  [3,5],
  [4,6],
  [5,7]
])

#Target salary (Actual Data)
y=np.array([23,26,28,33])

#Assumed Learned weights manually

w1=2
w2=3
b=1

#Prediction

predictions=w1*x[:,0] + w2*x[:,1] + b

#Calculate MSE
mse=np.mean((y - predictions)**2)
print("Predictions :", predictions)
print("Mean Squared Error:",mse)


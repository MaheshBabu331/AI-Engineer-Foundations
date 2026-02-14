import numpy as np

#Linear Regression Model From Scratch

#Step 1: Data 

x=np.array([1,2,3,4,5])
y=np.array([3,5,7,9,11])

#step 2 : Calculate Means
mean_x=np.mean(x)
mean_y=np.mean(y)

#Step 3: Calculate Slope (m)

numerator=np.sum((x - mean_x) * (y - mean_y))
denominator=np.sum((x - mean_x) ** 2)
m=numerator / denominator 

#Step 4: Calculate Intercept (b)

b = mean_y - m * mean_x

#Step 5: Predictions

predicted=m * x + b

#Step 6: Calculate Error (MSE)

mse= np.mean((y-predicted)**2)

print("Slope (M):", m)
print("Intercept (B):", b)
print("Predictions:", predicted)
print("Mean Squared Error:", mse)

#Predict salary for 6 years of experience

new_experience=6
new_salary=m*new_experience + b
print("Predicted salary for 6 years:", new_salary)


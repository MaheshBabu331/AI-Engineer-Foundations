import numpy as np
#feature scaling in numpy

data=np.array([
  [25,50000,672],
  [30,60000,800],
  [32,70000,720],
  [40,55000,640],
  [45,90000,780]
], dtype=float)
print("Original Data: \n", data)

#Min - Max salary

salary=data[:,1]
min_salary= np.min(salary)
max_salary=np.max(salary)
data[:,1]=(salary-min_salary)/(max_salary-min_salary)

#Standardization Age

age=data[:,0]
mean_age=np.mean(age)
std_age=np.std(age)
data[:,0]=(age-mean_age)/std_age
print("\n Scaled Data: \n",data)


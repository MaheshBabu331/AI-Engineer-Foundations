import numpy as np
#Broadcasting in Numpy
# we need to give bonus 2000 to each employee
salaries=np.array([12000,22000,40000])
updated_salaries=salaries+2000
print("Original Salaries:",salaries)
print("Updated Salaries:",updated_salaries)

#conditional Operations
#marks of some random students
marks=np.array([20,40,60,70,90,40,80])
#we need to update 10 marks who is getting below 50
marks[marks <50] +=10
print("Updated_Marks", marks)

#Data simumalations (mini ML Style)

data=np.array([
  [20,40000,650],
  [30,45000,500],
  [45,60000,700],
  [44,70000,780]
])
#Normalize salary into lakhs
data[:, 1] /= 100000
#Reduced credit score who are having less than 600 to reduce 20

data[data[:,2] < 600 , 2] -= 20

#Cap salary  at 1.0 (after normalization)

data[data[:, 1] > 1.0 , 1] = 1.0
print("Proceed Data \n", data)



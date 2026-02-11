import numpy as np
numbers=[10,20,30,40,50,60]
np_array=np.array(numbers)
print("python list:",numbers)
print("numpy arrays:",np_array)
print("shape:",np_array.shape)
print("Data type:",np_array.dtype)
zeros_array=np.zeros(6)
ones_array=np.ones(6)
range_array=np.arange(1,7)
print("Zeros:",zeros_array)
print("Ones:",ones_array)
print("Range:",range_array)
number=np.array([10,20,30,40,50,60])
print("Add 5:",number + 5)
print("multiple by 2:", number * 2)
print("First Element :", number[0])
print("Last two elements :", number[-2:])

# 2D array

matrix=np.array([
  [1,2,3],
  [4,5,6]
])
print("\n Matrix:")
print(matrix)
print("Shape :", matrix.shape)
print("Coloum_Wise:", np.sum(matrix, axis=0))
print("Row_Wise:", np.sum(matrix,axis=1))

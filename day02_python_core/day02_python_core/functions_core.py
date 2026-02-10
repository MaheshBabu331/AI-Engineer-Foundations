def calculate_average(numbers):
  if numbers==[]:
    raise ValueError("List is empty.cannot find average")
  total=sum(numbers)
  count=len(numbers)
  average=total/count
  return average
def divide(a,b):
  if b==0:
    raise ZeroDivisionError("cannot divide zero")
  result=a/b
  return result
numbers_list=[10,20,30,40,50]
try:
  avg=calculate_average(numbers_list)
  print("average:",avg)
  div=divide(10,3)
  print("division:",div)
except Exception as error:
  print("Error:",error)
  
  

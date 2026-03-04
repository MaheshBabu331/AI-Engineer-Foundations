from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
data=load_iris()
x=data.data
y=data.target
# first split : train and temp

x_train,x_temp,y_train,y_temp=train_test_split(x,y,test_size=0.3,random_state=42)

#Second split: validation and test
x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,test_size=0.3,random_state=42)
print("Train Size:",len(x_train))
print("Validation Size:", len(x_val))
print("Test Size:", len(x_test))



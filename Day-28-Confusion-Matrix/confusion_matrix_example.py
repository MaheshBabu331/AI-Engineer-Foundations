from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
data=load_iris()
x=data.data
y=data.target
x_train,y_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=SVC()
model.fit(x_train,y_train)
pred=model.predict(x_test)
cm=confusion_matrix(pred,y_test)
print("Confusion Matrix:")
print(cm)


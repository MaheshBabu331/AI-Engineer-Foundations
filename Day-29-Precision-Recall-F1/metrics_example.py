from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score,recall_score,f1_score
data=load_iris()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=SVC()
model.fit(x_train,y_train)
pred=model.predict(x_test)

#Evaluation Metrics
precision=precision_score(y_test,pred,average="macro")
recall=recall_score(y_test,pred,average="macro")
f1=f1_score(y_test,pred,average="macro")

print("Precision:",precision)
print("Recall:",recall)
print("F1 Score:", f1)

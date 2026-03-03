from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data=load_breast_cancer()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model=LogisticRegression(max_iter=5000)
model.fit(x_train,y_train)
y_prob=model.predict_proba(x_test)[:,1]
auc_score=roc_auc_score(y_test,y_prob)
print("AUC Score :",auc_score)

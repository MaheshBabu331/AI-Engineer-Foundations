from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

data=load_iris()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model= SVC()
param_grid={
  "C":[0.1,1,10],
  "kernel":["linear","rbf"]
}
grid=GridSearchCV(model,param_grid,cv=5)
grid.fit(x_train,y_train)
print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)
accuracy=grid.score(x_test,y_test)
print("Accuracy:", accuracy)

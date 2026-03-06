from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data=load_iris()
x=data.data
y=data.target

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)
pipeline=Pipeline([
  ("scaler", StandardScaler()),
  ("model",LogisticRegression())
])

pipeline.fit(x_train,y_train)
accuracy=pipeline.score(x_test,y_test)
print("Model Accuracy",accuracy)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Simple Binary Classification Dataset
x=np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y=np.array([0,0,0,0,1,1,1,1])

#Model 
model=LogisticRegression()
model.fit(x,y)
#Predictions
predictions=model.predict(x)
probabilities=model.predict_proba(x)
print("Model Predictions: ", predictions)
print("Probabilites :", probabilities)
print("Accuarcy :", accuracy_score(y,predictions))

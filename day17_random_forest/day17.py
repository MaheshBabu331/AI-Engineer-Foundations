import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Random Forest Classification

x=np.array([[1],[2],[3],[4],[5]
           [6],[7],[8]
           ])
y=np.array([0,0,0,0,0,1,1,1,1,1])
#Model
model=RandomForestClassifier(
  n_estimators=100,
  max_depth=2,
  random_state=42
)
model.fit(x,y)
predictions=model.predict(x)
print("Predictions :",predictions)
print("Accuracy Score :", accuracy_score(y,predictions))
  

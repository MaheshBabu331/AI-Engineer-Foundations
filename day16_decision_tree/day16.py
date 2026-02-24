import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Decision Tree Classification

#sample data
#Let us assume Features -->Hours studied
#result --> pass(1) or fail(0)
x=np.array([[2],[4],[6],[5]])
y=np.array([0,0,1,1])

#Model 
model=DecisionTreeClassifier(max_depth=1)
model.fit(x,y)

#Predictions 
predictions=model.predict(x)
print("Predictions :", predictions)
print("Accuracy:", accuracy_score(y, predictions))


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# train model
X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier()
model.fit(X, y)

# save model
joblib.dump(model, "model.pkl")

# load model
loaded_model = joblib.load("model.pkl")

# predict
print(loaded_model.predict([[5.1, 3.5, 1.4, 0.2]]))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# load dataset
data = load_iris()
X = data.data
y = data.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create pipeline
pipeline = Pipeline([
("scaler", StandardScaler()),
("model", SVC())
])

# train model
pipeline.fit(X_train, y_train)

# prediction
accuracy = pipeline.score(X_test, y_test)

print("Accuracy:", accuracy)

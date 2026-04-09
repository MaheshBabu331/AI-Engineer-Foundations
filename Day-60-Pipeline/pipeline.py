from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# data
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([20000, 30000, 40000, 50000, 60000])

# pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# train
pipeline.fit(X, y)

# predict
print(pipeline.predict([[6]]))

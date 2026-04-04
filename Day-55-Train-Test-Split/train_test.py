import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# data
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([22000, 28000, 35000, 45000, 52000, 60000, 68000, 75000])

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train
model = LinearRegression()
model.fit(X_train, y_train)

# test
y_pred = model.predict(X_test)

# evaluate
print("MAE:", int(mean_absolute_error(y_test, y_pred)))

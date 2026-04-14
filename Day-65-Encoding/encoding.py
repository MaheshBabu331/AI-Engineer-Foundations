from sklearn.preprocessing import OneHotEncoder
import numpy as np

X = np.array([['Male'], ['Female'], ['Male']])

encoder = OneHotEncoder(sparse=False)
X_new = encoder.fit_transform(X)

print(X_new)

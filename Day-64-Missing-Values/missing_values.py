import numpy as np
from sklearn.impute import SimpleImputer

X = np.array([[1],[2],[np.nan],[4]])

imputer = SimpleImputer(strategy='mean')
X_new = imputer.fit_transform(X)

print(X_new)

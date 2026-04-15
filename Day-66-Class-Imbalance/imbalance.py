from imblearn.over_sampling import SMOTE
import numpy as np

X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,0,1,1])  # imbalance

smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

print("Before:", y)
print("After:", y_res)

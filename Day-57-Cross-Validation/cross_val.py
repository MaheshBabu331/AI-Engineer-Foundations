from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([22000, 28000, 35000, 45000, 52000, 60000, 68000, 75000])

model = LinearRegression()

scores = cross_val_score(model, X, y, cv=5)

print("Scores:", scores)
print("Average Score:", scores.mean())

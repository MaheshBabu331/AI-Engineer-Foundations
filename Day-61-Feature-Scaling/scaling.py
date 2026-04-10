from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[25, 50000],
              [30, 60000],
              [35, 70000]])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)

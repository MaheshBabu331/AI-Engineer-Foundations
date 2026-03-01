from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

data = load_iris()
X = data.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])

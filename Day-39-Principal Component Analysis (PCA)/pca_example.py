from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

data = load_iris()
X = data.data

# reduce to 2 features
pca = PCA(n_components=2)

X_new = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_new.shape)

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif

data = load_iris()

X = data.data
y = data.target

# select top 2 features
selector = SelectKBest(score_func=f_classif, k=2)

X_new = selector.fit_transform(X, y)

print("Selected Features Shape:", X_new.shape)

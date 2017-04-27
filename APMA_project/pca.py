import pickle as pkl
from sklearn.decomposition import PCA

with open("X.pkl",'rb') as f:
    X = pkl.load(f)

with open("y.pkl",'rb') as f:
    y = pkl.load(f)

pca = PCA(n_components=100)
pca.fit(X)
print(sum(pca.explained_variance_ratio_))
X = pca.transform(X)


with open("PCA_X.pkl",'wb') as f:
    pkl.dump(X,f)
with open("PCA_model.pkl",'wb') as f:
    pkl.dump(pca, f)

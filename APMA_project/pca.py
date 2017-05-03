# This program is responsible of performing PCA on the image vectors after being flattened

# import necessary libraries
import pickle as pkl
from sklearn.decomposition import PCA

# obtain the image vectors after flattening them in 'face_flattening.py'
with open("X.pkl", 'rb') as f:
    X = pkl.load(f)

with open("y.pkl", 'rb') as f:
    y = pkl.load(f)

pca = PCA(n_components=100)     # perform pca on X, keeping 100 components
pca.fit(X)

# show the total percentage of variance explained by all selected components
print(sum(pca.explained_variance_ratio_))
X = pca.transform(X)     # apply dimensionality reduction to X

# save X and the pca model as .pkl files
with open("PCA_X.pkl", 'wb') as f:
    pkl.dump(X, f)
with open("PCA_model.pkl",'wb') as f:
    pkl.dump(pca, f)

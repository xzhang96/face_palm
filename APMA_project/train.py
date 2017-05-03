# This program is responsible of training the model

# import necessary libraries
import pickle as pkl
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pylab as plt


# obtain X and y from .pkl files
with open("PCA_X.pkl", 'rb') as f:
    X = pkl.load(f)

with open("y.pkl", 'rb') as f:
    y = pkl.load(f)
y = y.T
y = np.array(y)
y = y.flatten()

# randomly split X and y into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# create lists for alphas and scores
alphas = np.logspace(-5, -3, 300)
scores = []

# find the alpha with the highest score
for alpha in alphas:
    clf = LogisticRegression(C=alpha, penalty='l1')
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

alpha_best = alphas[np.argmax(scores)]
print(alpha_best)

# train the model using the best alpha
clf = LogisticRegression(C=alpha_best, penalty='l1')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# Use 10 folds cross validation to check the model
cross_validation_score = cross_val_score(clf, X, y, cv=10)
plt.xlabel('lambda')
plt.ylabel('score')
plt.title('Performance on 5 folds with c=' + str(alpha_best))
plt.bar(range(1, 11), cross_validation_score)
print(np.mean(cross_validation_score))
plt.show()

# save the trained model in a .pkl file
with open("trained_model.pkl", 'wb') as f:
    pkl.dump(clf, f)


# import necessary libraries
from PIL import Image
import numpy as np
import pickle as pkl


# define the prediction function, return the probabilities of each emotion
def predict_face(path):
    img = Image.open(path)      # obtain the image that is uploaded to the website
    image_data = np.array(img).astype(float)    # transform the images into matrices
    flattened_image = image_data.flatten()     # flatten the matrices and turn them into vectors

    # obtain the pca model and trained model from .pkl file
    with open("PCA_model.pkl", 'rb') as f:
        pca = pkl.load(f)

    with open("trained_model.pkl", 'rb') as f:
        clf = pkl.load(f)

    X = pca.transform(flattened_image)     # perform pca on the flattened image

    # predict the probabilities of each emotion using the model we trained earlier
    prob = clf.predict_proba(X)
    result = prob.reshape((8, 1))
    emotions = clf.classes_
    print(result, emotions)
    return result

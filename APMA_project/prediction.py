from PIL import Image
import numpy as np
import pickle as pkl


def predict_face(path):
    img = Image.open(path)
    image_data = np.array(img).astype(float)
    flattened_image = image_data.flatten()

    with open("PCA_model.pkl", 'rb') as f:
        pca = pkl.load(f)

    with open("trained_model.pkl", 'rb') as f:
        clf = pkl.load(f)

    X = pca.transform(flattened_image)
    prob = clf.predict_proba(X)
    result = prob.reshape((8, 1))
    emotions = clf.classes_
    print(result, emotions)
    print(type(result))
    return result

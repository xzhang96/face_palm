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
    result = clf.predict_proba(X)
    emotions = clf.classes_
    #print(result, file=open('result_prob.txt', 'w'))
    print(result, emotions)

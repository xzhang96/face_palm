from PIL import Image
import numpy as np
import glob
import pandas as pd
import pickle as pkl


emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotion order
emotion_list = []
image_list = []

for emotion in emotions:
    images = glob.glob("dataset/%s/*" % emotion)  # Get list of all images with emotion

    for image in images:
        emt = image.split('/')[1]
        img = Image.open(image)
        image_data = np.array(img).astype(float)
        flattened_image = image_data.flatten()
        emotion_list.append(emt)
        image_list.append(flattened_image)

y = pd.DataFrame(emotion_list)
X = pd.DataFrame(image_list)

print(y.shape)
print(X.shape)

with open("X.pkl",'wb') as f:
    pkl.dump(X, f)

with open("y.pkl",'wb') as f:
    pkl.dump(y, f)



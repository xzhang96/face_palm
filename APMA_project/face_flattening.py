# This program is responsible of flattening the images in the data set

# import necessary libraries
from PIL import Image
import numpy as np
import glob
import pandas as pd
import pickle as pkl


emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotion order
emotion_list = []
image_list = []

for emotion in emotions:
    images = glob.glob("dataset/%s/*" % emotion)  # Get list of all image path with emotion

    for image in images:
        emt = image.split('/')[1]  # extract the emotion name from the path of the image
        img = Image.open(image)   # open image using the path of the image
        image_data = np.array(img).astype(float)   # transform the image into a matrix
        flattened_image = image_data.flatten()     # flatten the matrix into a vector
        emotion_list.append(emt)
        image_list.append(flattened_image)


# construct pandas dataframe using the emotions and the image vectors
y = pd.DataFrame(emotion_list)
X = pd.DataFrame(image_list)

print(y.shape)
print(X.shape)

# save X and y as .pkl files
with open("X.pkl",'wb') as f:
    pkl.dump(X, f)

with open("y.pkl",'wb') as f:
    pkl.dump(y, f)



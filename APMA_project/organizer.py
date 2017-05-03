# This program is used to organize the image data we got from the dataset

# import necessary libraries
import glob
from shutil import copyfile
import os

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotion order
participants = glob.glob("source_emotion/*")  # Returns a list of all folders with participant numbers

for participant in participants:
    part = "%s" % participant[-4:]  # store current participant number
    for sessions in glob.glob("%s/*" % participant):  # Store list of sessions for current participant
        for files in glob.glob("%s/*" % sessions):
            current_session = files[20:-30]
            file = open(files, 'r')

            emotion = int(
                float(file.readline()))  # emotions are encoded as a float, readline as float, then convert to integer.

            # get path for last image in sequence, which contains the emotion
            sourcefile_emotion = glob.glob("source_images/%s/%s/*" % (part, current_session))[-1]
            # do the same thing for neutral image
            sourcefile_neutral = glob.glob("source_images/%s/%s/*" % (part, current_session))[0]

            # Generate path to put neutral image
            destination_neutral = "sorted_set/neutral/%s" % sourcefile_neutral[24:]
            # Do same for emotion containing image
            destination_emotion = "sorted_set/%s/%s" % (emotions[emotion], sourcefile_emotion[24:])

            copyfile(sourcefile_neutral, destination_neutral)  # Copy file
            copyfile(sourcefile_emotion, destination_emotion)  # Copy file

# clean up the "neutral" folder. One neutral image for each subject.
neutrals = glob.glob("sorted_set/neutral/*")

exist = {}
for neutral in neutrals:
    filename = neutral.split('/')[2]
    num = filename[0:3]
    if num not in exist:
        exist[num] = 1
    else:
        os.remove(neutral)







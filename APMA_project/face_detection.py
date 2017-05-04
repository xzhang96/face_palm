# This program is responsible for detecting faces from the images uploaded from the website

# import necessary libraries
import cv2


def detect_faces(image,filename):
    faceDet = cv2.CascadeClassifier("cv_classifier/haarcascade_frontalface_default.xml")
    faceDet2 = cv2.CascadeClassifier("cv_classifier/haarcascade_frontalface_alt2.xml")
    faceDet3 = cv2.CascadeClassifier("cv_classifier/haarcascade_frontalface_alt.xml")
    faceDet4 = cv2.CascadeClassifier("cv_classifier/haarcascade_frontalface_alt_tree.xml")

    frame = cv2.imread(image)  # Open image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    # Detect face using 4 different classifiers
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
    face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face2) == 1:
        facefeatures == face2
    elif len(face3) == 1:
        facefeatures = face3
    elif len(face4) == 1:
        facefeatures = face4
    else:
        facefeatures = ""

    # Cut and save face
    for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
        gray = gray[y:y + h, x:x + w]  # Cut the frame to size

        try:
            out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
            cv2.imwrite("static/"+filename+"face.jpg", out)  # Write image
        except:
            pass  # If error, pass file

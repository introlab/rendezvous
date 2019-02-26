import cv2
import sys
import os

workingDirectory = os.path.dirname(os.path.realpath(sys.argv[0]))

# Takes an image as input, returns coordinates of the faces detected in an array
# return format: [x, y, width, height]
def detectFaces(frame):
    # Create the haar cascade
    cascadePath = os.path.join(workingDirectory, 'HaarCascadeFiles/haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # Convert frame to grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return(faces)
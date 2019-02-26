# Code from https://github.com/shantnu/FaceDetect
import cv2
import sys

# Spawns a window that displays the frame with squares around faces
def showFrame(frame):
    cv2.imshow("Faces found", frame)
    cv2.waitKey(0)

def detectFaces(frame):
    # Create the haar cascade
    cascPath = '/home/walid/projet/rendezvous/src/FaceDetection/HaarCascadeFiles/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Convert frame to grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    showFrame(frame)

framePath = sys.argv[1]
image = cv2.imread(framePath)
detectFaces(image)

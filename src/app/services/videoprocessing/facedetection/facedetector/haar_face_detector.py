import os
from pathlib import Path

import cv2

from .iface_detector import IFaceDetector
from src.utils.rect import Rect

rootDirectory = str(Path(__file__).resolve().parents[6])


class HaarFaceDetector(IFaceDetector):

    def __init__(self):
        # Parameters for OpenCV Haar cascade detection
        self.cascadePath = os.path.join(rootDirectory, 'config/facedetection/haarcascade_frontalface_default.xml')
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)


    def detectFaces(self, image):
        (imageHeight, imageWidth) = image.shape[:2]

        # Convert frame to grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        validFaces = []
        for face in faces:
            (x, y, h, w) = face
            if x >= 0 and x <= imageWidth and y >= 0 and y <= imageHeight:
                validFaces.append(Rect(x + w / 2, y + h / 2, w, h))

        return validFaces


    def getName(self):
        return "Haar"
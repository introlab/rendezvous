import os
from pathlib import Path

import numpy as np
import cv2

from .iface_detector import IFaceDetector
from src.utils.rect import Rect

rootDirectory = str(Path(__file__).resolve().parents[6])


class DnnFaceDetector(IFaceDetector):

    def __init__(self):
        self.modelFile = os.path.join(rootDirectory, "config/facedetection/opencv_face_detector_uint8.pb")
        self.configFile = os.path.join(rootDirectory, "config/facedetection/opencv_face_detector.pbtxt")
        self.net = cv2.dnn.readNetFromTensorflow(self.modelFile, self.configFile)
        self.probabilityThreshold = 0.7


    def detectFaces(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (400, 400), [104, 117, 123], False, False)

        self.net.setInput(blob)
        detections = self.net.forward()

        numDetections = 0
        faces = np.zeros(detections.shape[2], dtype=object)
        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the probability
            probability = detections[0, 0, i, 2]
        
            # Filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if probability > self.probabilityThreshold:
                # Compute the (x, y) coordinates of the bounding box for the object
                box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
                (x1, y1, x2, y2) = box

                if x1 >= 0 and x2 <= w and y1 >= 0 and y2 <= h:
                    width = x2 - x1
                    height = y2 - y1
                    faces[numDetections] = Rect(x1 + width / 2, y1 + height / 2, width, height)
                    numDetections += 1
        
        return faces[:numDetections]


    def getName(self):
        return "Dnn"
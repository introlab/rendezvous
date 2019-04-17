import os
from pathlib import Path

#from pydarknet import Detector, Image

from .iface_detector import IFaceDetector
from src.utils.rect import Rect

rootDirectory = str(Path(__file__).resolve().parents[6])


class YoloFaceDetector(IFaceDetector):

    def __init__(self):
        self.cfg = os.path.join(rootDirectory, "config/yolo/cfg/yolov3-tiny.cfg")
        self.weights = os.path.join(rootDirectory, "config/yolo/weights/yolov3-tiny.weights")
        self.data = os.path.join(rootDirectory, "config/yolo/cfg/coco.data")
        self.net = Detector(bytes(self.cfg, encoding="utf-8"), bytes(self.weights, encoding="utf-8"), 0,
                            bytes(self.data, encoding="utf-8"))
        self.probabilityThreshold = 0.5


    def detectFaces(self, image):
        darkFrame = Image(image)
        results = self.net.detect(darkFrame)
        del darkFrame

        faces = []
        for cat, score, bounds in results:
            if score > self.probabilityThreshold:
                x, y, w, h = bounds
                faces.append(Rect(x, y, w, h))

        return faces


    def getName(self):
        return "Yolo"
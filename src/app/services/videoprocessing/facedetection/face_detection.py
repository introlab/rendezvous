import multiprocessing
import queue
import time

from .facedetector.dnn_face_detector import DnnFaceDetector


class FaceDetection(multiprocessing.Process):

    def __init__(self, imageQueue, facesQueue):
        super(FaceDetection, self).__init__()
        self.requestImage = True
        self.imageQueue = imageQueue
        self.facesQueue = facesQueue
        self.faceDetector = DnnFaceDetector()
        self.exit = multiprocessing.Event()


    def stop(self):
        self.exit.set()


    def run(self):
        print("Starting face detection")

        numberOfImagesProcessed = 0
        faces = []

        while not self.exit.is_set():

            frame = []
            try:
                frame = self.imageQueue.get_nowait()
                self.requestImage = False
            except queue.Empty:
                time.sleep(0.01)

            if frame != []:
                faces.extend(self.faceDetector.detectFaces(frame))
                numberOfImagesProcessed += 1
                if numberOfImagesProcessed == 3:
                    self.facesQueue.put(faces)
                    numberOfImagesProcessed = 0
                    faces = []
                    self.requestImage = True

        print("Face detection terminated")

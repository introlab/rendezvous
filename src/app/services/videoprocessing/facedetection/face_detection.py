import multiprocessing
import queue
import time

from .facedetector.dnn_face_detector import DnnFaceDetector


class FaceDetection(multiprocessing.Process):

    def __init__(self, imageQueue, facesQueue, heartbeatQueue, faceDetectionSemaphore, dewarpCount):
        super(FaceDetection, self).__init__()
        self.requestImage = True
        self.imageQueue = imageQueue
        self.facesQueue = facesQueue
        self.heartbeatQueue = heartbeatQueue
        self.faceDetectionSemaphore = faceDetectionSemaphore
        self.dewarpCount = dewarpCount
        self.faceDetector = DnnFaceDetector()
        self.exit = multiprocessing.Event()


    def stop(self):
        self.exit.set()


    def run(self):
        print("Starting face detection")

        dewarpIndex = 0
        faces = []

        lastHeartBeat = time.perf_counter()

        while not self.exit.is_set() and time.perf_counter() - lastHeartBeat < 0.5:

            image = None
            try:
                image, dewarpIndex = self.imageQueue.get_nowait()
                if dewarpIndex == 0:
                    self.faceDetectionSemaphore.acquire()
            except queue.Empty:
                time.sleep(0.01)

            if image is not None:

                imageFaces = self.faceDetector.detectFaces(image)
                if len(imageFaces) != 0:
                    faces.append((imageFaces, dewarpIndex))

                if dewarpIndex == self.dewarpCount - 1:
                    self.faceDetectionSemaphore.release()
                    if len(faces) != 0:
                        self.facesQueue.put(faces)
                        faces = []
            
            try:
                self.heartbeatQueue.get_nowait()
                lastHeartBeat = time.perf_counter()
            except queue.Empty:
                pass

        if dewarpIndex != 0:
            self.faceDetectionSemaphore.release()

        print("Face detection terminated")
        
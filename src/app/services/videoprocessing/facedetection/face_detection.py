import multiprocessing
import queue
import time

from .facedetector.dnn_face_detector import DnnFaceDetector


class FaceDetection(multiprocessing.Process):

    def __init__(self, imageQueue, facesQueue, heartbeatQueue, semaphore, dewarpCount):
        super(FaceDetection, self).__init__()
        self.requestImage = True
        self.imageQueue = imageQueue
        self.facesQueue = facesQueue
        self.heartbeatQueue = heartbeatQueue
        self.semaphore = semaphore
        self.dewarpCount = dewarpCount
        self.faceDetector = DnnFaceDetector()
        self.exit = multiprocessing.Event()


    def stop(self):
        self.exit.set()


    def run(self):
        print("Starting face detection")

        numberOfImagesProcessed = 0
        faces = []

        lastHeartBeat = time.perf_counter()

        while not self.exit.is_set() and time.perf_counter() - lastHeartBeat < 0.5:

            frame = None
            try:
                frame = self.imageQueue.get_nowait()
                if numberOfImagesProcessed == 0:
                    self.semaphore.acquire()
            except queue.Empty:
                time.sleep(0.01)

            if frame is not None:
                faces.extend(self.faceDetector.detectFaces(frame))

                numberOfImagesProcessed += 1
                if numberOfImagesProcessed == self.dewarpCount:
                    self.semaphore.release()
                    numberOfImagesProcessed = 0
                    if faces != []:
                        self.facesQueue.put(faces)
                        faces = []
            
            try:
                self.heartbeatQueue.get_nowait()
                lastHeartBeat = time.perf_counter()
            except queue.Empty:
                pass

        print("Face detection terminated")
        
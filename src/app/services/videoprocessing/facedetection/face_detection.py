from threading import Thread
import multiprocessing
import queue
import time

from .facedetector.yolo_face_detector import YoloFaceDetector


class FaceDetection(Thread):

    def __init__(self, imageQueue, facesQueue, heartbeatQueue, semaphore):
        super(FaceDetection, self).__init__()
        self.requestImage = True
        self.imageQueue = imageQueue
        self.facesQueue = facesQueue
        self.heartbeatQueue = heartbeatQueue
        self.semaphore = semaphore
        self.faceDetector = YoloFaceDetector()
        self.exit = multiprocessing.Event()


    def stop(self):
        self.exit.set()


    def run(self):
        print("Starting face detection")

        lastHeartBeat = time.perf_counter()

        while not self.exit.is_set() and time.perf_counter() - lastHeartBeat < 0.5:
            
            frame = []
            try:
                frame = self.imageQueue.get_nowait()
                self.semaphore.acquire()
            except queue.Empty:
                time.sleep(0.01)

            if frame != []:
                faces = self.faceDetector.detectFaces(frame)
                self.facesQueue.put(faces)
                self.semaphore.release()

            try:
                self.heartbeatQueue.get_nowait()
                lastHeartBeat = time.perf_counter()
            except queue.Empty:
                pass

        print("Face detection terminated")
        
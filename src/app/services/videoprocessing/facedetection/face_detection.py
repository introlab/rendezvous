import multiprocessing
import queue
import time

from .facedetector.dnn_face_detector import DnnFaceDetector


class FaceDetection(multiprocessing.Process):

    def __init__(self, imageQueue, facesQueue, semaphore):
        super(FaceDetection, self).__init__()
        self.requestImage = True
        self.imageQueue = imageQueue
        self.facesQueue = facesQueue
        self.semaphore = semaphore
        self.faceDetector = DnnFaceDetector()
        self.exit = multiprocessing.Event()


    def stop(self):
        self.exit.set()


    def run(self):
        print("Starting face detection")

        while not self.exit.is_set():

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

            time.sleep(0.01)


        print("Face detection terminated")

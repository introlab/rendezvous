import multiprocessing
import queue
import time

from .facedetector.yolo_face_detector import YoloFaceDetector
from .facedetector.dnn_face_detector import DnnFaceDetector
from .facedetector.haar_face_detector import HaarFaceDetector
from .facedetector.face_detection_methods import FaceDetectionMethods


class FaceDetection(multiprocessing.Process):

    def __init__(self, faceDetectionMethod, imageQueue, facesQueue, heartbeatQueue, semaphore):
        super(FaceDetection, self).__init__()
        self.requestImage = True
        self.imageQueue = imageQueue
        self.facesQueue = facesQueue
        self.heartbeatQueue = heartbeatQueue
        self.semaphore = semaphore
        self.faceDetectionMethod = faceDetectionMethod
        self.exit = multiprocessing.Event()


    def stop(self):
        self.exit.set()


    def run(self):
        print('Starting face detection')

        faceDetector = self.__createFaceDetector(self.faceDetectionMethod)

        lastHeartBeat = time.perf_counter()

        while not self.exit.is_set() and time.perf_counter() - lastHeartBeat < 0.5:
            
            frame = []
            try:
                frame = self.imageQueue.get_nowait()
                self.semaphore.acquire()
            except queue.Empty:
                time.sleep(0.01)

            if frame != []:
                faces = faceDetector.detectFaces(frame)
                self.facesQueue.put(faces)
                self.semaphore.release()

            try:
                self.heartbeatQueue.get_nowait()
                lastHeartBeat = time.perf_counter()
            except queue.Empty:
                pass

        print('Face detection terminated')

    
    def __createFaceDetector(self, faceDetectionMethod):
        if faceDetectionMethod == FaceDetectionMethods.OPENCV_DNN.value:
            return DnnFaceDetector()
        elif faceDetectionMethod == FaceDetectionMethods.OPENCV_HAAR_CASCADES.value:
            return HaarFaceDetector()
        elif faceDetectionMethod == FaceDetectionMethods.YOLO_V3.value:
            return YoloFaceDetector()
        else:
            return HaarFaceDetector()        
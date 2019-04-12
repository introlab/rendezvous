import multiprocessing
import queue
import time

from .facedetector.yolo_face_detector import YoloFaceDetector
from .facedetector.dnn_face_detector import DnnFaceDetector
from .facedetector.haar_face_detector import HaarFaceDetector
from .facedetector.face_detection_methods import FaceDetectionMethods


class FaceDetection(multiprocessing.Process):

    def __init__(self, faceDetectionMethod, imageQueue, facesQueue, heartbeatQueue, isBusySemaphore):
        super(FaceDetection, self).__init__()
        self.faceDetectionMethod = faceDetectionMethod
        self.imageQueue = imageQueue
        self.facesQueue = facesQueue
        self.heartbeatQueue = heartbeatQueue
        self.isBusySemaphore = isBusySemaphore
        self.exit = multiprocessing.Event()
        self.requestImage = True
        self.isBusy = False


    def stop(self):
        self.exit.set()


    def run(self):
        print('Starting face detection')

        try:

            self.__lockOnce()

            faceDetector = self.__createFaceDetector(self.faceDetectionMethod)

            dewarpIndex = -1
            faces = []

            lastHeartBeat = time.perf_counter()

            while not self.exit.is_set() and time.perf_counter() - lastHeartBeat < 0.5:

                image = None
                try:
                    image, dewarpIndex = self.imageQueue.get_nowait()
                    self.__lockOnce()
                except queue.Empty:
                    self.__unlockOnce()
                    time.sleep(0.01)

                if image is not None:
                    imageFaces = faceDetector.detectFaces(image)
                    self.facesQueue.put((dewarpIndex, imageFaces))
                
                try:
                    self.heartbeatQueue.get_nowait()
                    lastHeartBeat = time.perf_counter()
                except queue.Empty:
                    pass

        finally:
            
            self.__unlockOnce()
        
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


    def __lockOnce(self):
        if not self.isBusy:
            self.isBusySemaphore.acquire()
            self.isBusy = True

    
    def __unlockOnce(self):
        if self.isBusy:
            self.isBusySemaphore.release()
            self.isBusy = False
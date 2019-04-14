import multiprocessing
import queue
import time

from src.app.services.service.process.generic_process import GenericProcess
from src.utils.exception_helper import ExceptionHelper
from .facedetector.yolo_face_detector import YoloFaceDetector
from .facedetector.dnn_face_detector import DnnFaceDetector
from .facedetector.haar_face_detector import HaarFaceDetector
from .facedetector.face_detection_methods import FaceDetectionMethods


class FaceDetection(GenericProcess):

    def __init__(self, faceDetectionMethod):
        super(FaceDetection, self).__init__()
        self.faceDetectionMethod = faceDetectionMethod
        self.imageQueue = multiprocessing.Queue()
        self.facesQueue = multiprocessing.Queue()
        self.isBusySemaphore = multiprocessing.Semaphore()
        self.requestImage = True
        self.isBusy = False


    def stop(self):
        super(FaceDetection, self).stop()
        self.emptyQueue(self.facesQueue)
        self.emptyQueue(self.imageQueue)


    def run(self):
        print('Starting face detection')

        try:

            self.aquireLockOnce()

            faceDetector = self.__createFaceDetector(self.faceDetectionMethod)

            dewarpIndex = -1
            faces = []

            lastKeepAliveTimestamp = time.perf_counter()

            while not self.exit.is_set() and time.perf_counter() - lastKeepAliveTimestamp < 0.5:

                image = None
                try:
                    dewarpIndex, image = self.imageQueue.get_nowait()
                    self.aquireLockOnce()
                except queue.Empty:
                    self.releaseLockOnce()
                    time.sleep(0.01)

                if image is not None:
                    imageFaces = faceDetector.detectFaces(image)
                    self.facesQueue.put((dewarpIndex, imageFaces))
                
                try:
                    self.keepAliveQueue.get_nowait()
                    lastKeepAliveTimestamp = time.perf_counter()
                except queue.Empty:
                    pass
        
        except Exception as e:
            ExceptionHelper.printStackTrace(e)
            self.exceptionQueue.put(e)

        finally:
            self.releaseLockOnce()
            print('Face detection terminated')


    def tryGetFaces(self):
        try:                  
            return self.facesQueue.get_nowait()
        except queue.Empty:
            return None


    def sendDewarpedImages(self, dewarpIndex, buffer):
        self.imageQueue.put_nowait((dewarpIndex, buffer))


    def aquireLockOnce(self, block = True):
        if not self.isBusy:
            self.isBusy = self.isBusySemaphore.acquire(block)

        return self.isBusy

    
    def releaseLockOnce(self):
        if self.isBusy:
            self.isBusySemaphore.release()
            self.isBusy = False


    def __createFaceDetector(self, faceDetectionMethod):
        if faceDetectionMethod == FaceDetectionMethods.OPENCV_DNN.value:
            return DnnFaceDetector()
        elif faceDetectionMethod == FaceDetectionMethods.OPENCV_HAAR_CASCADES.value:
            return HaarFaceDetector()
        elif faceDetectionMethod == FaceDetectionMethods.YOLO_V3.value:
            return YoloFaceDetector()
        else:
            return HaarFaceDetector()
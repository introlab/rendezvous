import multiprocessing
import queue
import time

#from .facedetector.yolo_face_detector import YoloFaceDetector
from .facedetector.dnn_face_detector import DnnFaceDetector
from .facedetector.haar_face_detector import HaarFaceDetector
from .facedetector.face_detection_methods import FaceDetectionMethods


class FaceDetection(multiprocessing.Process):

    def __init__(self, faceDetectionMethod, imageQueue, facesQueue, heartbeatQueue, detectFacesSemaphore, dewarpCount):
        super(FaceDetection, self).__init__()
        self.faceDetectionMethod = faceDetectionMethod
        self.imageQueue = imageQueue
        self.facesQueue = facesQueue
        self.heartbeatQueue = heartbeatQueue
        self.detectFacesSemaphore = detectFacesSemaphore
        self.dewarpCount = dewarpCount
        self.faceDetector = DnnFaceDetector()
        self.exit = multiprocessing.Event()
        self.requestImage = True


    def stop(self):
        self.exit.set()


    def run(self):
        print('Starting face detection')

        faceDetector = self.__createFaceDetector(self.faceDetectionMethod)

        dewarpIndex = -1
        faces = []

        lastHeartBeat = time.perf_counter()

        while not self.exit.is_set() and time.perf_counter() - lastHeartBeat < 0.5:

            image = None
            try:
                image, dewarpIndex = self.imageQueue.get_nowait()
                if dewarpIndex == 0:
                    self.detectFacesSemaphore.acquire()
            except queue.Empty:
                time.sleep(0.01)

            if image is not None:

                imageFaces = self.faceDetector.detectFaces(image)
                if len(imageFaces) != 0:
                    faces.append((imageFaces, dewarpIndex))

                if dewarpIndex == self.dewarpCount - 1:
                    self.detectFacesSemaphore.release()
                    if len(faces) != 0:
                        self.facesQueue.put(faces)
                        faces = []
            
            try:
                self.heartbeatQueue.get_nowait()
                lastHeartBeat = time.perf_counter()
            except queue.Empty:
                pass

        if dewarpIndex != -1 and dewarpIndex != self.dewarpCount - 1:
            self.detectFacesSemaphore.release()
        
        print('Face detection terminated')

    
    def __createFaceDetector(self, faceDetectionMethod):
        if faceDetectionMethod == FaceDetectionMethods.OPENCV_DNN.value:
            return DnnFaceDetector()
        elif faceDetectionMethod == FaceDetectionMethods.OPENCV_HAAR_CASCADES.value:
            return HaarFaceDetector()
        #elif faceDetectionMethod == FaceDetectionMethods.YOLO_V3.value:
        #    return YoloFaceDetector()
        else:
            return HaarFaceDetector()        

import time
from math import radians
from multiprocessing import Queue, Semaphore
from threading import Thread
import queue

from PyQt5.QtCore import QObject, pyqtSignal

from .streaming.video_stream import VideoStream
from .virtualcamera.virtual_camera_manager import VirtualCameraManager
from src.utils.file_helper import FileHelper
from .facedetection.face_detection import FaceDetection
from .facedetection.facedetector.face_detection_methods import FaceDetectionMethods


class VideoProcessor(QObject):

    signalFrameData = pyqtSignal(object, object)
    signalStateChanged = pyqtSignal(bool)
    signalException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(VideoProcessor, self).__init__(parent)
        self.virtualCameraManager = VirtualCameraManager()
        self.isRunning = False
        self.imageQueue = Queue()
        self.facesQueue = Queue()
        self.semaphore = Semaphore()
        self.heartbeatQueue = Queue(1)


    def getCameraParams(self):
        cameraParams = {
            'fisheyeAngle': radians(self.cameraConfig['Image']['FisheyeAngle']),
            'baseDonutSlice': self.baseDonutSlice,
            'dewarpingParameters': self.dewarpingParameters
        }

        return cameraParams


    def start(self, cameraConfigPath, faceDetectionMethod):
        print("Starting video processor...")

        try:
            
            if not cameraConfigPath:
                raise Exception('cameraConfigPath needs to be set in the settings')

            if not faceDetectionMethod in [fdMethod.value for fdMethod in FaceDetectionMethods]:
                raise Exception('{} is not a supported face detection method'.format(self.faceDetectionMethod))

            Thread(target=self.run, args=(cameraConfigPath, faceDetectionMethod)).start()

        except Exception as e:
            
            self.signalException.emit(e)


    def stop(self):
         self.isRunning = False


    def run(self, cameraConfigPath, faceDetectionMethod):

        videoStream = None
        faceDetection = None

        try:

            self.cameraConfig = FileHelper.readJsonFile(cameraConfigPath)
            videoStream = VideoStream(self.cameraConfig)
            videoStream.initializeStream()

            self.baseDonutSlice = videoStream.getBaseDonutSlice()
            self.dewarpingParameters = videoStream.getDewarpingParameters()

            faceDetection = FaceDetection(faceDetectionMethod, self.imageQueue, self.facesQueue, self.heartbeatQueue, self.semaphore)
            faceDetection.start()

            print('Video processor started')

            self.isRunning = True
            self.signalStateChanged.emit(True)
            
            prevTime = time.perf_counter()
            while self.isRunning:

                try:
                    self.heartbeatQueue.put_nowait(True)
                except queue.Full:
                    pass
                
                currentTime = time.perf_counter()
                frameTime = currentTime - prevTime

                newFaces = None
                try:                  
                    newFaces = self.facesQueue.get_nowait()
                except queue.Empty:
                    pass

                success, frame = videoStream.readFrame()
                frameHeight, frameWidth, colors = frame.shape

                if self.semaphore.acquire(False):
                    self.semaphore.release()
                    self.imageQueue.put_nowait(frame.copy())

                if newFaces is not None:
                    self.virtualCameraManager.updateFaces(newFaces, frameWidth, frameHeight)

                if success:
                    self.virtualCameraManager.update(frameTime, frameWidth, frameHeight)
                    self.signalFrameData.emit(frame.copy(), self.virtualCameraManager.getVirtualCameras())
                    
                prevTime = currentTime
                
        except Exception as e:
            self.signalException.emit(e)

        finally:

            if faceDetection:
                faceDetection.stop()
                faceDetection.join()
                faceDetection = None

            self.__emptyQueue(self.imageQueue)
            self.__emptyQueue(self.facesQueue)
            self.__emptyQueue(self.heartbeatQueue)

            if videoStream:
                videoStream.destroy()
                videoStream = None

            self.signalStateChanged.emit(False)

        print('Video stream terminated')


    def __emptyQueue(self, queue):

        try:
            while True:
                queue.get_nowait()
        except:
            pass


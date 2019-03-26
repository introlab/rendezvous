import queue
import time
from multiprocessing import Queue
from multiprocessing import Semaphore
from threading import Thread

from PyQt5.QtCore import QObject, pyqtSignal

from .streaming.video_stream import VideoStream
from .facedetection.face_detection import FaceDetection
from .virtualcamera.virtual_camera_manager import VirtualCameraManager
from src.utils.file_helper import FileHelper


class VideoProcessor(QObject):

    signalFrameData = pyqtSignal(object, object)
    signalException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(VideoProcessor, self).__init__(parent)
        self.virtualCameraManager = VirtualCameraManager()
        self.isRunning = False
        self.imageQueue = Queue()
        self.facesQueue = Queue()
        self.semaphore = Semaphore()
        self.heartbeatQueue = Queue(1)


    def getCameraParams():
        cameraParams = []
        cameraParams['fisheyeAngle'] = self.cameraConfig.FisheyeAngle
        cameraParams['baseDonutSlice'] = self.baseDonutSlice
        cameraParams['dewarpingParameters'] = self.dewarpingParameters

        return cameraParams

    def start(self, cameraConfigPath):
        print("Starting video processor...")

        try:
                               
            if not cameraConfigPath:
                raise Exception('cameraConfigPath needs to be set in the settings')

            Thread(target=self.run, args=(cameraConfigPath,)).start()

        except Exception as e:
            
            self.isRunning = False
            self.signalException.emit(e)


    def stop(self):
         self.isRunning = False


    def run(self, cameraConfigPath):

        videoStream = None
        faceDetection = None

        try:

            cameraConfig = FileHelper.readJsonFile(cameraConfigPath)
            videoStream = VideoStream(cameraConfig)
            videoStream.initializeStream()
             
            self.baseDonutSlice = videoStream.getBaseDonutSlice(videoStream)
            self.dewarpingParameters = videoStream.getDewarpingParameters(videoStream)
            
            self.baseDonutSlice = videoStream.getBaseDonutSlice(videoStream)
            self.dewarpingParameters = videoStream.getDewarpingParameters(videoStream)

            faceDetection = FaceDetection(self.imageQueue, self.facesQueue, self.heartbeatQueue, self.semaphore)
            faceDetection.start()

            print('Video processor started')

            prevTime = time.perf_counter()
            self.isRunning = True
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

            self.isRunning = False
            self.signalException.emit(e)

        finally:

            if faceDetection:
                faceDetection.stop()
                faceDetection.join()
                faceDetection.terminate()
                faceDetection = None

            if videoStream:
                videoStream.destroy()

        print('Video stream terminated')

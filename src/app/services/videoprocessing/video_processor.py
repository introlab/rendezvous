import queue
import time
from multiprocessing import Queue
from multiprocessing import Semaphore
from threading import Thread
from collections import deque

from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np

from .streaming.video_stream import VideoStream
from .facedetection.face_detection import FaceDetection
from .virtualcamera.virtual_camera_manager import VirtualCameraManager
from src.utils.file_helper import FileHelper
from src.app.services.videoprocessing.streaming.camera_config import CameraConfig
from src.utils.dewarping_helper import DewarpingHelper
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import DonutSlice
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import FisheyeDewarping


class VideoProcessor(QObject):

    signalFrameData = pyqtSignal(object)
    signalException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(VideoProcessor, self).__init__(parent)
        self.virtualCameraManager = VirtualCameraManager()
        self.isRunning = False
        self.imageQueue = Queue()
        self.facesQueue = Queue()
        self.semaphore = Semaphore()
        self.heartbeatQueue = Queue(1)


    def start(self, cameraConfigPath):
        print("Starting video processor...")

        try:
            
            if not cameraConfigPath:
                raise Exception('cameraConfigPath needs to be set in the settings')

            Thread(target=self.run, args=(cameraConfigPath,)).start()

            self.isRunning = True

        except Exception as e:
            
            self.isRunning = False
            self.signalException.emit(e)


    def stop(self):
         self.isRunning = False


    def run(self, cameraConfigPath):

        videoStream = None
        faceDetection = None

        try:

            cameraConfig = CameraConfig(FileHelper.readJsonFile(cameraConfigPath))

            videoStream = VideoStream(cameraConfig)
            videoStream.initializeStream()

            success, frame = videoStream.readFrame()

            if (not success):
                raise Exception("Failed to read image and retrieve the number of channels")

            channels = len(frame.shape)

            dewarpCount = 4
            fdDewarpingParameters = self.__getFaceDetectionDewarpingParameters(cameraConfig, dewarpCount)
            vcDewarpingParameters = self.__getVirtualCameraDewarpingParameters(cameraConfig)
            
            fdOutputWidth = int((fdDewarpingParameters[0].dewarpWidth / 4) - (fdDewarpingParameters[0].dewarpWidth / 4) % 4)
            fdOutputHeight = int((fdDewarpingParameters[0].dewarpHeight / 4) - (fdDewarpingParameters[0].dewarpHeight / 4) % 4)
            vcOutputWidth = 600
            vcOutputHeight = 800

            dewarper = FisheyeDewarping(cameraConfig.imageWidth, cameraConfig.imageHeight, channels, True)

            #fdBuffer = np.empty((fdOutputHeight, fdOutputWidth, channels), dtype=np.uint8)
            fdBufferId = dewarper.createRenderContext(fdOutputWidth, fdOutputHeight, channels)

            #vcBuffer = np.empty((vcOutputHeight, vcOutputWidth, channels), dtype=np.uint8)
            vcBufferId = dewarper.createRenderContext(vcOutputWidth, vcOutputHeight, channels)

            faceDetection = FaceDetection(self.imageQueue, self.facesQueue, self.heartbeatQueue, self.semaphore, dewarpCount)
            faceDetection.start()

            fdBufferQueue = deque()
            vcBufferQueue = deque()
            
            print('Video processor started')

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

                if newFaces is not None:
                    self.virtualCameraManager.updateFaces(newFaces, fdOutputWidth, fdOutputHeight)

                self.virtualCameraManager.update(frameTime, fdOutputWidth, fdOutputHeight)
                
                success, frame = videoStream.readFrame()

                if success:

                    dewarper.loadFisheyeImage(frame)

                    if self.semaphore.acquire(False):
                        self.semaphore.release()
                        for i in range(0, dewarpCount):
                            fdBuffer = np.empty((fdOutputHeight, fdOutputWidth, channels), dtype=np.uint8)
                            dewarper.queueDewarping(fdBufferId, fdDewarpingParameters[i], fdBuffer)
                            fdBufferQueue.append(fdBuffer)

                    for vc in self.virtualCameraManager.getVirtualCameras():
                        vcBuffer = np.empty((vcOutputHeight, vcOutputWidth, channels), dtype=np.uint8)
                        # Generate dewarping params for vc
                        dewarper.queueDewarping(vcBufferId, vcDewarpingParameters, vcBuffer)
                        vcBufferQueue.append(vcBuffer)

                    bufferId = 0
                    while bufferId != -1:
                        bufferId = dewarper.dewarpNextImage()
                        
                        if bufferId == fdBufferId:
                            frameHeight, frameWidth, colors = fdBuffer.shape
                            buffer = fdBufferQueue.pop()
                            self.imageQueue.put_nowait(buffer)

                    self.signalFrameData.emit(vcBufferQueue)

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

            self.__emptyQueue(self.imageQueue)
            self.__emptyQueue(self.facesQueue)
            self.__emptyQueue(self.heartbeatQueue)

            if videoStream:
                videoStream.destroy()
                videoStream = None

            self.virtualCameraManager.clear()

        print('Video stream terminated')


    def __getFaceDetectionDewarpingParameters(self, cameraConfig, dewarpCount):
        dewarpingParameters = []
        donutSlice = DonutSlice(cameraConfig.imageWidth / 2, cameraConfig.imageHeight / 2, cameraConfig.inRadius, \
            cameraConfig.outRadius, np.deg2rad(0), np.deg2rad(cameraConfig.angleSpan))

        for i in range(0, dewarpCount):
            dewarpingParameters.append(DewarpingHelper.getDewarpingParameters(donutSlice, \
                cameraConfig.topDistorsionFactor, cameraConfig.bottomDistorsionFactor))
            donutSlice.middleAngle = (donutSlice.middleAngle + np.deg2rad(360 / dewarpCount)) % (2 * np.pi)

        return dewarpingParameters


    def __getVirtualCameraDewarpingParameters(self, cameraConfig):
        donutSlice = DonutSlice(cameraConfig.imageWidth / 2, cameraConfig.imageHeight / 2, cameraConfig.inRadius, \
            cameraConfig.outRadius, np.deg2rad(0), np.deg2rad(cameraConfig.angleSpan))

        return DewarpingHelper.getDewarpingParameters(donutSlice, \
                cameraConfig.topDistorsionFactor, cameraConfig.bottomDistorsionFactor)

                
    def __emptyQueue(self, queue):

        try:
            while True:
                queue.get_nowait()
        except:
            pass

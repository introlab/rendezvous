import time
from math import radians
from multiprocessing import Queue, Semaphore
import queue
from threading import Thread
from collections import deque

from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np

from .streaming.video_stream import VideoStream
from .virtualcamera.virtual_camera_manager import VirtualCameraManager
from .virtualcamera.face import Face
from src.utils.file_helper import FileHelper
from src.utils.spherical_angles_rect import SphericalAnglesRect
from src.utils.dewarping_helper import DewarpingHelper
from src.utils.spherical_angles_converter import SphericalAnglesConverter
from src.app.services.videoprocessing.streaming.camera_config import CameraConfig
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import DonutSlice
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import FisheyeDewarping
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import NoQueuedDewarping
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import NoDewarpingRead
from .facedetection.face_detection import FaceDetection
from .facedetection.facedetector.face_detection_methods import FaceDetectionMethods


class VideoProcessor(QObject):

    signalVirtualCameras = pyqtSignal(object, object)
    signalStateChanged = pyqtSignal(bool)
    signalException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(VideoProcessor, self).__init__(parent)
        self.virtualCameraManager = VirtualCameraManager(0, np.pi * 2)
        self.isRunning = False
        self.imageQueue = Queue()
        self.facesQueue = Queue()
        self.isBusySemaphore = Semaphore()
        self.heartbeatQueue = Queue(1)
        self.fpsTarget = 15


    def start(self, cameraConfigPath, faceDetectionMethod):
        print("Starting video processor...")

        try:
            
            if not cameraConfigPath:
                raise Exception('cameraConfigPath needs to be set in the settings tab')

            if not faceDetectionMethod in [fdMethod.value for fdMethod in FaceDetectionMethods]:
                raise Exception('Unsupported face detection method: {}. Set a correct method in the settings tab.'.format(faceDetectionMethod))

            Thread(target=self.run, args=(cameraConfigPath, faceDetectionMethod)).start()

        except Exception as e:
            
            self.signalException.emit(e)


    def stop(self):
         self.isRunning = False


    def run(self, cameraConfigPath, faceDetectionMethod):

        videoStream = None
        dewarper = None
        faceDetection = None
        show360DegAsVcs = False

        try:

            cameraConfig = CameraConfig(FileHelper.readJsonFile(cameraConfigPath))

            videoStream = VideoStream(cameraConfig)
            videoStream.initializeStream()

            success, frame = videoStream.readFrame()

            if not success:
                raise Exception('Failed to read image and retrieve the number of channels')

            if frame.shape[1] != cameraConfig.imageWidth or frame.shape[0] != cameraConfig.imageHeight:
                raise Exception('Camera image does\'t have same size as the config, perhaps the port number is wrong')

            channels = frame.shape[2]

            dewarpCount = 4
            fdDewarpingParameters = self.__getFaceDetectionDewarpingParametersList(cameraConfig, dewarpCount)
            
            # Increasing these will make face detection slower, but yield better detections
            fdOutputWidth = int((fdDewarpingParameters[0].dewarpWidth / 2) - (fdDewarpingParameters[0].dewarpWidth / 2) % 4)
            fdOutputHeight = int((fdDewarpingParameters[0].dewarpHeight / 2) - (fdDewarpingParameters[0].dewarpHeight / 2) % 4)

            vcOutputWidth = 300
            vcOutputHeight = 400

            fisheyeCenter = (cameraConfig.imageWidth / 2, cameraConfig.imageHeight / 2)
            fisheyeAngle = np.deg2rad(cameraConfig.fisheyeAngle)

            dewarper = FisheyeDewarping()

            fdFisheyeBufferId = dewarper.createFisheyeContext(cameraConfig.imageWidth, cameraConfig.imageHeight, channels)
            fisheyeBufferId = dewarper.createFisheyeContext(cameraConfig.imageWidth, cameraConfig.imageHeight, channels)

            fdBufferId = dewarper.createRenderContext(fdOutputWidth, fdOutputHeight, channels)
            vcBufferId = dewarper.createRenderContext(vcOutputWidth, vcOutputHeight, channels)

            faceDetection = FaceDetection(faceDetectionMethod, self.imageQueue, \
                self.facesQueue, self.heartbeatQueue, self.isBusySemaphore)
            faceDetection.start()

            # Make sure the face detection is started before starting video stream
            time.sleep(0.1)
            self.isBusySemaphore.acquire()
            self.isBusySemaphore.release()

            fdBufferQueue = deque()

            fdBuffers = []
            for dewarpIndex in range(0, dewarpCount):
                fdBuffers.append(np.zeros((fdOutputHeight, fdOutputWidth, channels), dtype=np.uint8))

            angleFaces = []
            
            print('Video processor started')

            self.isRunning = True
            self.signalStateChanged.emit(True)

            frameTimeDelta = 0
            frameTimeGoal = 1./self.fpsTarget
            actualFrameTime = 1./self.fpsTarget
            currentTime = time.perf_counter()

            while self.isRunning:

                try:
                    self.heartbeatQueue.put_nowait(True)
                except queue.Full:
                    pass

                newFaces = None
                try:                  
                    newFaces = self.facesQueue.get_nowait()
                except queue.Empty:
                    pass

                if newFaces is not None:
                    (dewarpIndex, faces) = newFaces

                    for face in faces:
                        angleFace = self.__getSphericalAnglesRectFromFace(face, fdDewarpingParameters[dewarpIndex], \
                            fdOutputWidth, fdOutputHeight, fisheyeAngle, fisheyeCenter)
                        angleFaces.append(angleFace)

                    if dewarpIndex == dewarpCount - 1:
                        self.virtualCameraManager.updateFaces(angleFaces)
                        angleFaces = []

                self.virtualCameraManager.update(actualFrameTime)

                success, frame = videoStream.readFrame()
                readTime = time.perf_counter() - currentTime

                if success and readTime < frameTimeGoal / 2:

                    vcBuffers = []

                    # By default the fisheye image is loaded in the fisheyeBufferId, 
                    # for face detection it's loaded to fdFisheyeBufferId.
                    currentFisheyeBufferId = fisheyeBufferId

                    # Create all buffers required for face detection dewarping
                    if not fdBufferQueue and readTime < frameTimeGoal / 4 and self.isBusySemaphore.acquire(False) :
                        self.isBusySemaphore.release()
                        currentFisheyeBufferId = fdFisheyeBufferId
                        for dewarpIndex in range(0, dewarpCount):
                            fdBuffer = np.empty((fdOutputHeight, fdOutputWidth, channels), dtype=np.uint8)
                            fdBufferQueue.append((fdBuffer, dewarpIndex))

                    dewarper.loadFisheyeImage(currentFisheyeBufferId, frame)

                    loadTime = time.perf_counter() - currentTime

                    if loadTime < (3 * frameTimeGoal) / 4:

                        # Queue next dewarping for face detection (only one dewarping per frame)
                        if fdBufferQueue and readTime < frameTimeGoal / 4:
                            fdBuffer, dewarpIndex = fdBufferQueue[0]
                            # Face detection doesn't need te most recent image, it needs to use the same image for all indexes
                            dewarper.queueDewarping(fdFisheyeBufferId, fdBufferId, fdDewarpingParameters[dewarpIndex], fdBuffer)

                        # Queue dewarping for each virtual camera
                        for vc in self.virtualCameraManager.getVirtualCameras():
                            vcBuffer = np.empty((vcOutputHeight, vcOutputWidth, channels), dtype=np.uint8)

                            # Generate dewarping params for vc
                            vcDewarpingParameters = self.__getVirtualCameraDewarpingParameters(vc, fisheyeCenter, cameraConfig)
    
                            # Virtual camera always need the most recent fisheye image
                            dewarper.queueDewarping(currentFisheyeBufferId, vcBufferId, vcDewarpingParameters, vcBuffer)
                            vcBuffers.append(vcBuffer)

                        # Execute all queued dewarping for face detection and virtual cameras
                        bufferId = 0
                        while bufferId != NoQueuedDewarping:
                            bufferId = dewarper.dewarpNextImage()
                            
                            if bufferId == fdBufferId:
                                buffer, dewarpIndex = fdBufferQueue.popleft()
                                fdBuffers[dewarpIndex] = buffer
                                self.imageQueue.put_nowait((buffer, dewarpIndex))
                        
                        # Dislay 360 dewarping as virtual cameras for debugging
                        if show360DegAsVcs:
                            vcBuffers.extend(fdBuffers)

                        # Send the virtual camera images
                        self.signalVirtualCameras.emit(vcBuffers, self.virtualCameraManager.getVirtualCameras())

                frameTime = time.perf_counter() - currentTime
                
                if frameTime < frameTimeGoal:
                    time.sleep(frameTimeGoal - frameTime)

                prevTime = currentTime
                currentTime = time.perf_counter()
                actualFrameTime = currentTime - prevTime

                frameTimeDelta += 1./self.fpsTarget - actualFrameTime

                if np.abs(frameTimeDelta) < 1./(self.fpsTarget * 6):
                    frameTimeGoal = 1./self.fpsTarget + frameTimeDelta
                else:
                    frameTimeGoal = 1./self.fpsTarget + 1./(self.fpsTarget * 6) * np.sign(frameTimeDelta)
                
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

            if dewarper:
                dewarper.cleanUp()
                
            self.virtualCameraManager.clear()
            self.signalStateChanged.emit(False)

        print('Video stream terminated')


    def __getFaceDetectionDewarpingParametersList(self, cameraConfig, dewarpCount):
        dewarpingParametersList = []
        donutSlice = DonutSlice(cameraConfig.imageWidth / 2, cameraConfig.imageHeight / 2, cameraConfig.inRadius, \
            cameraConfig.outRadius, np.deg2rad(0), np.deg2rad(cameraConfig.angleSpan))

        for i in range(0, dewarpCount):
            dewarpingParametersList.append(DewarpingHelper.getDewarpingParameters(donutSlice, \
                cameraConfig.topDistorsionFactor, cameraConfig.bottomDistorsionFactor))
            donutSlice.middleAngle = (donutSlice.middleAngle + np.deg2rad(360 / dewarpCount)) % (2 * np.pi)

        return dewarpingParametersList


    def __getVirtualCameraDewarpingParameters(self, virtualCamera, fisheyeCenter, cameraConfig):
        azimuth, elevation = virtualCamera.getMiddlePosition()
        azimuthSpan = virtualCamera.getAzimuthSpan()
        azimuthLeft, azimuthRight, elevationBottom, elevationTop = virtualCamera.getAngleCoordinates()

        donutSlice = DonutSlice(fisheyeCenter[0], fisheyeCenter[1], \
            cameraConfig.inRadius, cameraConfig.outRadius, azimuth, azimuthSpan)

        dewarpingParameters = DewarpingHelper.getDewarpingParameters(donutSlice, \
            cameraConfig.topDistorsionFactor, cameraConfig.bottomDistorsionFactor)

        maxElevation = SphericalAnglesConverter.getElevationFromImage(dewarpingParameters.dewarpWidth / 2, 0, \
            np.deg2rad(cameraConfig.fisheyeAngle), fisheyeCenter, dewarpingParameters)
        minElevation = SphericalAnglesConverter.getElevationFromImage(dewarpingParameters.dewarpWidth / 2, dewarpingParameters.dewarpHeight, \
            np.deg2rad(cameraConfig.fisheyeAngle), fisheyeCenter, dewarpingParameters)

        deltaElevation = maxElevation - minElevation
        deltaElevationTop = maxElevation - elevationTop
        deltaElevationBottom = elevationBottom - minElevation

        dewarpingParameters.topOffset = (deltaElevationTop * dewarpingParameters.dewarpHeight) / deltaElevation
        dewarpingParameters.bottomOffset = (deltaElevationBottom * dewarpingParameters.dewarpHeight) / deltaElevation

        return dewarpingParameters

    
    def __getSphericalAnglesRectFromFace(self, face, dewarpingParameters, outputWidth, outputHeight, fisheyeAngle, fisheyeCenter):
        (x1, y1, x2, y2) = face.getBoundingRect()
        xCenter, yCenter = face.getPosition()

        dewarpWidthFactor = dewarpingParameters.dewarpWidth / outputWidth
        dewarpHeightFactor = dewarpingParameters.dewarpHeight / outputHeight

        xNew1 = x1 * dewarpWidthFactor
        yNew1 = y1 * dewarpHeightFactor
        xNew2 = x2 * dewarpWidthFactor
        yNew2 = y2 * dewarpHeightFactor

        if xCenter > outputWidth / 2:
            xMostTop = xNew1
            xMostBottom = xNew2
            yMostLeft = yNew2
            yMostRight = yNew1
        else:
            xMostTop = xNew2
            xMostBottom = xNew1
            yMostLeft = yNew1
            yMostRight = yNew2

        azimuthLeft = SphericalAnglesConverter.getAzimuthFromImage(xNew1, yMostLeft, \
            fisheyeAngle, fisheyeCenter, dewarpingParameters)
        azimuthRight = SphericalAnglesConverter.getAzimuthFromImage(xNew2, yMostRight, \
            fisheyeAngle, fisheyeCenter, dewarpingParameters)
        elevationTop = SphericalAnglesConverter.getElevationFromImage(xMostTop, yNew1, \
            fisheyeAngle, fisheyeCenter, dewarpingParameters)
        elevationBottom = SphericalAnglesConverter.getElevationFromImage(xMostBottom, yNew2, \
            fisheyeAngle, fisheyeCenter, dewarpingParameters)

        return SphericalAnglesRect(azimuthLeft, azimuthRight, elevationBottom, elevationTop)
                
    def __emptyQueue(self, queue):

        try:
            while True:
                queue.get_nowait()
        except:
            pass


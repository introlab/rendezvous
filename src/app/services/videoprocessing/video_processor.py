from threading import Thread

from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np

from .streaming.video_stream import VideoStream
from .facedetection.facedetector.dnn_face_detector import DnnFaceDetector
from .virtualcamera.virtual_camera_manager import VirtualCameraManager
from src.utils.file_helper import FileHelper
from src.app.services.videoprocessing.streaming.camera_config import CameraConfig
from src.utils.dewarping_helper import DewarpingHelper
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import DonutSlice
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import FisheyeDewarping


class VideoProcessor(QObject):

    signalFrameData = pyqtSignal(object, object)
    signalVideoException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(VideoProcessor, self).__init__(parent)

        self.faceDetector = DnnFaceDetector()
        self.virtualCameraManager = VirtualCameraManager()
        self.isRunning = False


    # Set debug to true to show the areas of the calculations
    def start(self, cameraConfigPath):
        print("Starting video processor...")

        try:
                               
            if not cameraConfigPath:
                raise Exception('cameraConfigPath needs to be set in the settings')

            Thread(target=self.run, args=(cameraConfigPath,)).start()

        except Exception as e:
            
            self.isRunning = False
            self.signalVideoException.emit(e)


    def stop(self):
         self.isRunning = False


    def run(self, cameraConfigPath):

        videoStream = None

        try:

            cameraConfig = CameraConfig(FileHelper.readJsonFile(cameraConfigPath))

            videoStream = VideoStream(cameraConfig)
            videoStream.initializeStream()

            success, frame = videoStream.readFrame()

            if (not success):
                raise Exception("Failed to read image and retrieve the number of channels")

            channels = len(frame.shape)
            dewarpCount = 4

            dewarpingParameters = []
            donutSlice = DonutSlice(cameraConfig.imageWidth / 2, cameraConfig.imageHeight / 2, cameraConfig.inRadius, \
                cameraConfig.outRadius, np.deg2rad(0), np.deg2rad(cameraConfig.angleSpan))

            for i in range(0, dewarpCount):
                dewarpingParameters.append(DewarpingHelper.getDewarpingParameters(donutSlice, \
                    cameraConfig.topDistorsionFactor, cameraConfig.bottomDistorsionFactor))
                donutSlice.middleAngle = (donutSlice.middleAngle + np.deg2rad(360 / dewarpCount)) % (2 * np.pi)
            
            outputWidth = int((dewarpingParameters[0].dewarpWidth / 2) - (dewarpingParameters[0].dewarpWidth / 2) % 4)
            outputHeight = int((dewarpingParameters[0].dewarpHeight / 2) - (dewarpingParameters[0].dewarpHeight / 2) % 4)

            dewarper = FisheyeDewarping(cameraConfig.imageWidth, cameraConfig.imageHeight, channels, True)

            dewarpedImageBuffers = np.zeros((2, outputHeight, outputWidth, channels), dtype=np.uint8)
            dewarpedImageBuffersId = []
            dewarpedImageBuffersId.append(dewarper.bindDewarpingBuffer(dewarpedImageBuffers[0]))
            dewarpedImageBuffersId.append(dewarper.bindDewarpingBuffer(dewarpedImageBuffers[1]))
            dewarpedImageBuffersIndex = 0

            print('Video processor started')

            self.isRunning = True
            while self.isRunning:

                success, frame = videoStream.readFrame()

                if success:

                    dewarper.loadFisheyeImage(frame)

                    for i in range(0, 1):
                        dewarpedImageBuffersIndex = (dewarpedImageBuffersIndex + 1) % 2

                        dewarper.queueDewarping(dewarpedImageBuffersId[dewarpedImageBuffersIndex], dewarpingParameters[2])
                        dewarper.dewarpNextImage()
                        dewarpedFrame = dewarpedImageBuffers[dewarpedImageBuffersIndex]

                        faces = self.faceDetector.detectFaces(dewarpedFrame)

                        frameHeight, frameWidth, colors = dewarpedFrame.shape
                        self.virtualCameraManager.update(faces.tolist(), frameWidth, frameHeight)

                        self.signalFrameData.emit(dewarpedFrame, self.virtualCameraManager.getVirtualCameras())

        except Exception as e:

            self.isRunning = False
            self.signalVideoException.emit(e)

        finally:

            if videoStream:
                videoStream.destroy()

            self.virtualCameraManager.clear()

        print('Video stream terminated')


    def __dewarpFrame(self):
        pass


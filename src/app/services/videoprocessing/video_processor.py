import queue
import time
from multiprocessing import Queue
from threading import Thread

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

    signalFrameData = pyqtSignal(object, object)
    signalException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(VideoProcessor, self).__init__(parent)
        self.virtualCameraManager = VirtualCameraManager()
        self.isRunning = False
        self.imageQueue = Queue()
        self.facesQueue = Queue()


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

            dewarpedImageBuffer = np.zeros((outputHeight, outputWidth, channels), dtype=np.uint8)
            dewarpedImageBufferId = dewarper.createRenderContext(outputWidth, outputHeight, channels)

            faceDetection = FaceDetection(self.imageQueue, self.facesQueue)
            faceDetection.start()
            
            print('Video processor started')

            prevTime = time.perf_counter()
            self.isRunning = True
            while self.isRunning:

                currentTime = time.perf_counter()
                frameTime = currentTime - prevTime

                newFaces = None
                try:                  
                    newFaces = self.facesQueue.get_nowait()
                except queue.Empty:
                    time.sleep(0)

                success, frame = videoStream.readFrame()

                if success:

                    dewarper.loadFisheyeImage(frame)

                    for i in range(0, 1):

                        dewarper.queueDewarping(dewarpedImageBufferId, dewarpingParameters[2], dewarpedImageBuffer)
                        dewarper.dewarpNextImage()
                        dewarpedFrame = dewarpedImageBuffer

                        frameHeight, frameWidth, colors = dewarpedFrame.shape

                        if faceDetection.requestImage:
                            self.imageQueue.put_nowait(dewarpedFrame)

                        if newFaces is not None:
                            self.virtualCameraManager.updateFaces(newFaces, frameWidth, frameHeight)

                        
                        self.virtualCameraManager.update(frameTime, frameWidth, frameHeight)
                        self.signalFrameData.emit(dewarpedFrame.copy(), self.virtualCameraManager.getVirtualCameras())
                    
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

            self.virtualCameraManager.clear()

        print('Video stream terminated')

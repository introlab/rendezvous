from threading import Thread

from PyQt5.QtCore import QObject, pyqtSignal

from .streaming.video_stream import VideoStream
from .facedetection.facedetector.dnn_face_detector import DnnFaceDetector
from .virtualcamera.virtual_camera_manager import VirtualCameraManager
from src.utils.file_helper import FileHelper


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

            cameraConfig = FileHelper.readJsonFile(cameraConfigPath)
            videoStream = VideoStream(cameraConfig)
            videoStream.initializeStream()

            print('Video processor started')

            self.isRunning = True
            while self.isRunning:

                success, frame = videoStream.readFrame()

                if success:
                    faces = self.faceDetector.detectFaces(frame)

                    frameHeight, frameWidth, colors = frame.shape
                    self.virtualCameraManager.update(faces.tolist(), frameWidth, frameHeight)

                    self.signalFrameData.emit(frame, self.virtualCameraManager.getVirtualCameras())

        except Exception as e:

            self.isRunning = False
            self.signalVideoException.emit(e)

        finally:

            if videoStream:
                videoStream.destroy()

            self.virtualCameraManager.clear()

        print('Video stream terminated')


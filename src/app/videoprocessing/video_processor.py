from threading import Thread

from PyQt5.QtCore import QObject, pyqtSignal

from .streaming.video_stream import VideoStream
from .facedetection.facedetector.dnn_face_detector import DnnFaceDetector
from src.utils.file_helper import FileHelper


class VideoProcessor(QObject):

    signalFrameData = pyqtSignal(object, object)
    signalVideoException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(VideoProcessor, self).__init__(parent)

        self.faceDetector = DnnFaceDetector()
        self.isRunning = False


    # Set debug to true to show the areas of the calculations
    def start(self, debug, cameraConfigPath):
        print('Starting video processor...')

        try:
                               
            if not cameraConfigPath:
                raise Exception('cameraConfigPath needs to be set in the settings')

            Thread(target=self.run, args=(cameraConfigPath, debug)).start()

        except Exception as e:
            
            self.isRunning = False
            self.signalVideoException.emit(e)


    def stop(self):
         self.isRunning = False


    def run(self, cameraConfigPath, debug):

        videoStream = None

        try:

            cameraConfig = FileHelper.readJsonFile(cameraConfigPath)
            videoStream = VideoStream(cameraConfig, debug)
            videoStream.initializeStream()

            print('Video processor started')

            self.isRunning = True
            while self.isRunning:

                success, frame = videoStream.readFrame()

                if success:
                    faces = self.faceDetector.detectFaces(frame)
                    self.signalFrameData.emit(frame, faces)

        except Exception as e:

            self.isRunning = False
            self.signalVideoException.emit(e)

        finally:

            if videoStream:
                videoStream.destroy()

        print('Video stream terminated')


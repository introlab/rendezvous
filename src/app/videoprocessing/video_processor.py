import os
from threading import Thread

from PyQt5.QtCore import QObject, pyqtSignal

from .streaming.video_stream import VideoStream
from .facedetection.facedetector.dnn_face_detector import DnnFaceDetector


class VideoProcessor(QObject):

    signalFrameData = pyqtSignal(object, object)

    def __init__(self, cameraConfig, debug, parent=None):
        super(VideoProcessor, self).__init__(parent)

        self.faceDetector = DnnFaceDetector()

        self.videoStream = VideoStream(cameraConfig, debug)
        self.isRunning = False


    def start(self):
        print("Starting video processor...")
        Thread(target=self.run).start()


    def stop(self):
         self.isRunning = False


    # Initialize stream, displays the source and dewarped image, set debug to true to show the areas of the calculations
    def run(self):
        print("Video processor started")
        self.videoStream.initializeStream()
        self.isRunning = True
        while self.isRunning:

            success, frame = self.videoStream.readFrame()

            if success:
                faces = self.faceDetector.detectFaces(frame)
                self.signalFrameData.emit(frame, faces)

        self.videoStream.destroy()
        
        print("Video stream terminated")
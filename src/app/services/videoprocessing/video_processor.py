import queue
import time
from multiprocessing import Queue
from threading import Thread

from PyQt5.QtCore import QObject, pyqtSignal

from .streaming.video_stream import VideoStream
from .facedetection.face_detection import FaceDetection
from src.utils.file_helper import FileHelper


class VideoProcessor(QObject):

    signalFrameData = pyqtSignal(object, object)
    signalException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(VideoProcessor, self).__init__(parent)
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

            cameraConfig = FileHelper.readJsonFile(cameraConfigPath)
            videoStream = VideoStream(cameraConfig)
            videoStream.initializeStream()

            faceDetection = FaceDetection(self.imageQueue, self.facesQueue)
            faceDetection.start()

            print('Video processor started')

            self.isRunning = True
            faces = []
            while self.isRunning:
                newFaces = None
                try:
                    newFaces = self.facesQueue.get_nowait()
                except queue.Empty:
                    time.sleep(0)

                if newFaces is not None:
                    faces = newFaces

                success, frame = videoStream.readFrame()

                if faceDetection.requestImage:
                    self.imageQueue.put_nowait(frame)

                if success:
                    self.signalFrameData.emit(frame, faces)

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
                videoStream = None
                
        print('Video stream terminated')

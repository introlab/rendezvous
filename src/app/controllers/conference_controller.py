from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.app.services.odas.odas import Odas
from src.app.services.videoprocessing.video_processor import VideoProcessor
from src.app.services.sourceclassifier.source_classifier import SourceClassifier


class ConferenceController(QObject):

    signalException = pyqtSignal(Exception)
    signalAudioPositions = pyqtSignal(object)
    signalOdasState = pyqtSignal(bool)
    signalVideoProcessorState = pyqtSignal(bool)
    signalVirtualCamerasReceived = pyqtSignal(object, object)
    signalHumanSourcesDetected = pyqtSignal(object)

    def __init__(self, parent=None):
        super(ConferenceController, self).__init__(parent)

        self.__odas = Odas(hostIP='127.0.0.1', portPositions=10020, portAudio=10030, isVerbose=True)
        self.__odas.start()

        self.isRecording = False
        self.isOdasLiveConnected = False
        self.videoProcessorState = False

        self.__odas.signalException.connect(self.odasExceptionHandling)
        
        self.__odas.signalPositionData.connect(self.positionDataReceived)
        self.__odas.signalClientsConnected.connect(self.odasClientConnected)

        self.__videoProcessor = VideoProcessor()
        self.__videoProcessor.signalException.connect(self.videoProcessorExceptionHandling)
        self.__videoProcessor.signalFrameData.connect(self.virtualCamerasReceived)
        self.__videoProcessor.signalStateChanged.connect(self.videoProcessorStateChanged)

        self.__positions = {}


    @pyqtSlot(bool)
    def odasClientConnected(self, isConnected):
        self.isOdasLiveConnected = isConnected
        self.signalOdasState.emit(isConnected)


    @pyqtSlot(bool)
    def videoProcessorStateChanged(self, state):
        self.videoProcessorState = state
        self.signalVideoProcessorState.emit(state)


    @pyqtSlot(object)
    def positionDataReceived(self, positions):
        self.__positions = positions
        self.signalAudioPositions.emit(positions)


    @pyqtSlot(object, object)
    def virtualCamerasReceived(self, image, virtualCameras):
        if(self.__positions):
            # range threshold in degrees
            rangeThreshold = 15
            cameraParams = self.__videoProcessor.getCameraParams()
            sourceClassifier = SourceClassifier(cameraParams, rangeThreshold)
            sourceClassifier.classifySources(virtualCameras, self.__positions)
            self.signalHumanSourcesDetected.emit(sourceClassifier.humanSources)

        self.signalVirtualCamerasReceived.emit(image, virtualCameras)


    @pyqtSlot(Exception)
    def odasExceptionHandling(self, e):
        self.stopOdasLive()
        self.signalOdasState.emit(False)
        self.signalException.emit(e)

        
    @pyqtSlot(Exception)
    def videoProcessorExceptionHandling(self, e):
        self.stopVideoProcessor()
        self.signalVideoProcessorState.emit(False)
        self.signalException.emit(e)


    def startOdasLive(self, odasPath, micConfigPath):
        self.__odas.startOdasLive(odasPath, micConfigPath)


    def stopOdasLive(self):
        self.__odas.stopOdasLive()


    def stopOdasServer(self):
        if self.__odas.isRunning:
            self.__odas.stop()


    def startVideoProcessor(self, cameraConfigPath, faceDetection):
        if self.__videoProcessor and not self.__videoProcessor.isRunning:
            self.__videoProcessor.start(cameraConfigPath, faceDetection)


    def stopVideoProcessor(self):
        if self.__videoProcessor and self.__videoProcessor.isRunning:
            self.__videoProcessor.stop()


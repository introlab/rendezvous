from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.app.application_container import ApplicationContainer
from src.app.services.videoprocessing.video_processor import VideoProcessor
from src.app.services.sourceclassifier.source_classifier import SourceClassifier


class ConferenceController(QObject):

    signalException = pyqtSignal(Exception)
    signalOdasState = pyqtSignal(bool)
    signalVideoProcessorState = pyqtSignal(bool)
    signalVirtualCamerasReceived = pyqtSignal(object)
    signalHumanSourcesDetected = pyqtSignal(object)
    signalAudioPositions = pyqtSignal(object)

    def __init__(self, parent=None):
        super(ConferenceController, self).__init__(parent)

        self.isOdasLiveConnected = False
        self.videoProcessorState = False

        ApplicationContainer.odas().signalException.connect(self.odasExceptionHandling)
        ApplicationContainer.odas().signalPositionData.connect(self.positionDataReceived)
        ApplicationContainer.odas().signalClientsConnected.connect(self.odasClientConnected)

        self.__videoProcessor = VideoProcessor()
        self.__videoProcessor.signalException.connect(self.videoProcessorExceptionHandling)
        self.__videoProcessor.signalVirtualCameras.connect(self.virtualCamerasReceived)
        self.__videoProcessor.signalStateChanged.connect(self.videoProcessorStateChanged)

        self.__positions = {}


    @pyqtSlot(bool)
    def odasClientConnected(self, isConnected):
        self.isOdasLiveConnected = isConnected
        self.signalOdasState.emit(isConnected)


    @pyqtSlot(object)
    def positionDataReceived(self, positions):
        self.__positions = positions
        self.signalAudioPositions.emit(positions)


    @pyqtSlot(bool)
    def videoProcessorStateChanged(self, state):
        self.videoProcessorState = state
        self.signalVideoProcessorState.emit(state)


    @pyqtSlot(object, object)
    def virtualCamerasReceived(self, images, virtualCameras):
        if(self.__positions):
            # range threshold in degrees
            rangeThreshold = 15
            sourceClassifier = SourceClassifier(rangeThreshold)
            sourceClassifier.classifySources(virtualCameras, self.__positions)
            self.signalHumanSourcesDetected.emit(sourceClassifier.humanSources)

        self.signalVirtualCamerasReceived.emit(images)


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
        ApplicationContainer.odas().startOdasLive(odasPath, micConfigPath)


    def stopOdasLive(self):
        ApplicationContainer.odas().stopOdasLive()


    def stopOdasServer(self):
        if ApplicationContainer.odas().isRunning:
            ApplicationContainer.odas().stop()


    def startVideoProcessor(self, cameraConfigPath, faceDetection):
        if self.__videoProcessor and not self.__videoProcessor.isRunning:
            self.__videoProcessor.start(cameraConfigPath, faceDetection)


    def stopVideoProcessor(self):
        if self.__videoProcessor and self.__videoProcessor.isRunning:
            self.__videoProcessor.stop()


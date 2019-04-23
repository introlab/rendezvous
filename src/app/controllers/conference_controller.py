import time

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5 import QtGui

from src.app.application_container import ApplicationContainer
from src.app.services.service.service_state import ServiceState
from src.app.services.videoprocessing.virtualcamera.virtual_camera_display_builder import VirtualCameraDisplayBuilder


class ConferenceController(QObject):

    signalException = pyqtSignal(Exception)
    signalOdasState = pyqtSignal(bool)
    signalVideoProcessorState = pyqtSignal(bool)
    
    signalVirtualCamerasReceived = pyqtSignal(object)
    signalAudioPositions = pyqtSignal(object)

    def __init__(self, virtualCameraFrame, parent=None):
        super(ConferenceController, self).__init__(parent)

        self.__odasServer = ApplicationContainer.odas()

        self.__videoProcessor = ApplicationContainer.videoProcessor()

        self.__virtualCameraFrame = virtualCameraFrame

        self.__caughtOdasExceptions = []
        self.__caughtVideoExceptions = []
        self.__positions = {}


    def startOdasLive(self, odasPath, micConfigPath):
        if self.__odasServer.state != ServiceState.STOPPED:
            self.signalOdasState.emit(False)
            self.signalException.emit(Exception('Odas already started'))
            return

        self.__odasServer.signalStateChanged.connect(self.__odasStateChanged)
        self.__odasServer.startOdasLive(odasPath, micConfigPath)  


    def stopOdasLive(self):
        self.__odasServer.stopOdasLive()
            

    def startVideoProcessor(self, cameraConfigPath, faceDetectionMethod):
        if self.__videoProcessor.state != ServiceState.STOPPED:
            self.signalVideoProcessorState.emit(False)
            self.signalException.emit(Exception('Video already started'))
            return

        if self.__videoProcessor.state == ServiceState.STOPPED:
            self.__videoProcessor.signalStateChanged.connect(self.__videoProcessorStateChanged)
            self.__videoProcessor.start(cameraConfigPath, faceDetectionMethod)


    def stopVideoProcessor(self):
        self.__videoProcessor.stop()


    def close(self):
        if self.__odasServer.state == ServiceState.RUNNING:
            self.stopOdasLive()

        if self.__videoProcessor.state == ServiceState.RUNNING:
            self.stopVideoProcessor()

        if self.__odasServer.isRunning:
            self.__odasServer.stop()

        while self.__videoProcessor.state != ServiceState.STOPPED:
            time.sleep(0.01)

        self.__videoProcessor.destroy()


    @pyqtSlot(object)
    def __odasStateChanged(self, serviceState):
        self.__odasServer.state = serviceState

        if self.__odasServer.state == ServiceState.STARTING:
            self.__odasServer.signalException.connect(self.__odasExceptionHandling)

        elif self.__odasServer.state == ServiceState.RUNNING:
            self.__odasServer.signalPositionData.connect(self.__positionDataReceived)
            self.signalOdasState.emit(True)

        elif self.__odasServer.state == ServiceState.STOPPING:
            self.__odasServer.signalPositionData.disconnect(self.__positionDataReceived)

        elif self.__odasServer.state == ServiceState.STOPPED:
            self.__odasServer.signalException.disconnect(self.__odasExceptionHandling)
            self.__odasServer.signalStateChanged.disconnect(self.__odasStateChanged)
            for e in self.__caughtOdasExceptions:
                self.signalException.emit(e)
                self.__caughtOdasExceptions.clear()
            self.signalOdasState.emit(False)


    @pyqtSlot(object)
    def __videoProcessorStateChanged(self, serviceState):
        self.__videoProcessor.state = serviceState

        if self.__videoProcessor.state == ServiceState.STARTING:
            self.__videoProcessor.signalException.connect(self.__videoProcessorExceptionHandling)
            self.__videoProcessor.signalVirtualCameras.connect(self.__virtualCamerasReceived)

        if self.__videoProcessor.state == ServiceState.RUNNING:
            self.signalVideoProcessorState.emit(True)

        elif self.__videoProcessor.state == ServiceState.STOPPING:
            self.__videoProcessor.signalVirtualCameras.disconnect(self.__virtualCamerasReceived)

        elif self.__videoProcessor.state == ServiceState.STOPPED:
            self.__videoProcessor.signalException.disconnect(self.__videoProcessorExceptionHandling)
            self.__videoProcessor.signalStateChanged.disconnect(self.__videoProcessorStateChanged)
            for e in self.__caughtVideoExceptions:
                self.signalException.emit(e)
                self.__caughtVideoExceptions.clear()
            self.signalVideoProcessorState.emit(False)


    @pyqtSlot(object)
    def __positionDataReceived(self, positions):
        self.__positions = positions
        self.signalAudioPositions.emit(positions)


    @pyqtSlot(object, object)
    def __virtualCamerasReceived(self, images, virtualCameras):
        combinedImage = VirtualCameraDisplayBuilder.buildImage(images, self.__virtualCameraFrame.size(),
                                                                self.__virtualCameraFrame.palette().color(QtGui.QPalette.Background), 10)

        self.signalVirtualCamerasReceived.emit(combinedImage)


    @pyqtSlot(Exception)
    def __odasExceptionHandling(self, e):
        self.__caughtOdasExceptions.append(e)
        self.stopOdasLive()

        
    @pyqtSlot(Exception)
    def __videoProcessorExceptionHandling(self, e):
        self.__caughtVideoExceptions.append(e)
        self.stopVideoProcessor()


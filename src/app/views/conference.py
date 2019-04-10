from enum import Enum, unique
from math import degrees

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QFileDialog QWidget

from src.app.application_container import ApplicationContainer
from src.app.controllers.conference_controller import ConferenceController
from src.app.gui.conference_ui import Ui_Conference
from src.app.services.videoprocessing.virtualcamera.virtual_camera_displayer import VirtualCameraDisplayer


@unique
class BtnOdasLabels(Enum):
    START_ODAS = 'Start ODAS'
    STOP_ODAS = 'Stop ODAS'


@unique
class BtnVideoLabels(Enum):
    START_VIDEO = 'Start Video'
    STOP_VIDEO = 'Stop Video'


class Conference(QWidget, Ui_Conference):

    def __init__(self, odasserver, parent=None):
        super(Conference, self).__init__(parent)
        self.setupUi(self)
        self.conferenceController = ConferenceController(odasserver)

        self.virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)


        self.__isVideoSignalsConnected = False
        self.__isOdasSignalsConnected = False

        # Qt signal slots
        self.btnStartStopOdas.clicked.connect(self.btnStartStopOdasClicked)
        self.btnStartStopVideo.clicked.connect(self.btnStartStopVideoClicked)

        self.conferenceController.signalException.connect(self.exceptionReceived)
        self.conferenceController.signalHumanSourcesDetected.connect(self.showHumanSources)

        self.soundSources = []


    @pyqtSlot()
    def btnStartStopOdasClicked(self):
        self.btnStartStopOdas.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopOdas.text() == BtnOdasLabels.START_ODAS.value:
            self.conferenceController.signalAudioPositions.connect(self.positionDataReceived)
            self.conferenceController.signalOdasState.connect(self.odasStateChanged)
            self.__isOdasSignalsConnected = True
            self.conferenceController.startOdasLive(odasPath=ApplicationContainer.settings().getValue('odasPath'), micConfigPath=ApplicationContainer.settings().getValue('micConfigPath'))
        else:
            self.conferenceController.stopOdasLive()


    @pyqtSlot()
    def btnStartStopVideoClicked(self):
        self.btnStartStopVideo.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopVideo.text() == BtnVideoLabels.START_VIDEO.value:
            self.conferenceController.signalVideoProcessorState.connect(self.videoProcessorStateChanged)
            self.conferenceController.signalVirtualCamerasReceived.connect(self.updateVirtualCamerasDispay)
            self.__isVideoSignalsConnected = True
            self.conferenceController.startVideoProcessor(ApplicationContainer.settings().getValue('cameraConfigPath'), ApplicationContainer.settings().getValue('faceDetection'))
        else:
            self.conferenceController.stopVideoProcessor()


    @pyqtSlot(object)
    def positionDataReceived(self, values):

        self.source1AzimuthValueLabel.setText('%.5f' % degrees(values[0]['azimuth']))
        self.source2AzimuthValueLabel.setText('%.5f' % degrees(values[1]['azimuth']))
        self.source3AzimuthValueLabel.setText('%.5f' % degrees(values[2]['azimuth']))
        self.source4AzimuthValueLabel.setText('%.5f' % degrees(values[3]['azimuth']))

        self.source1ElevationValueLabel.setText('%.5f' % degrees(values[0]['elevation']))
        self.source2ElevationValueLabel.setText('%.5f' % degrees(values[1]['elevation']))
        self.source3ElevationValueLabel.setText('%.5f' % degrees(values[2]['elevation']))
        self.source4ElevationValueLabel.setText('%.5f' % degrees(values[3]['elevation']))

        self.soundSources = values


    @pyqtSlot(object)        
    def updateVirtualCamerasDispay(self, virtualCameraImages):
        self.virtualCameraDisplayer.updateDisplay(virtualCameraImages)


    @pyqtSlot(bool)
    def odasStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopOdas.setText(BtnOdasLabels.STOP_ODAS.value)
        
        else:
            self.btnStartStopOdas.setText(BtnOdasLabels.START_ODAS.value)
            self.conferenceController.signalAudioPositions.disconnect(self.positionDataReceived)
            self.conferenceController.signalOdasState.disconnect(self.odasStateChanged)
            self.__isOdasSignalsConnected = False

        self.btnStartStopOdas.setDisabled(False)

    
    @pyqtSlot(bool)
    def videoProcessorStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopVideo.setText(BtnVideoLabels.STOP_VIDEO.value)
            self.virtualCameraDisplayer.startDisplaying()
            
        else:
            self.btnStartStopVideo.setText(BtnVideoLabels.START_VIDEO.value)
            self.virtualCameraDisplayer.stopDisplaying()
            self.conferenceController.signalVideoProcessorState.disconnect(self.videoProcessorStateChanged)
            self.conferenceController.signalVirtualCamerasReceived.disconnect(self.updateVirtualCamerasDispay)
            self.__isVideoSignalsConnected = False

        self.btnStartStopVideo.setDisabled(False)


    @pyqtSlot(Exception)
    def exceptionReceived(self, e):
        if self.__isVideoSignalsConnected:
            self.conferenceController.signalVideoProcessorState.disconnect(self.videoProcessorStateChanged)
            self.conferenceController.signalVirtualCamerasReceived.disconnect(self.updateVirtualCamerasDispay)
        
        if self.__isOdasSignalsConnected:
            self.conferenceController.signalAudioPositions.disconnect(self.positionDataReceived)
            self.conferenceController.signalOdasState.disconnect(self.odasStateChanged)
        
        self.window().emitToExceptionsManager(e)


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.conferenceController.stopOdasServer()
            self.conferenceController.stopVideoProcessor()
            event.accept()
    

    def showHumanSources(self, humanSources):
        for index, source in enumerate(self.soundSources):
            if index in humanSources:
                self.__setSourceBackgroundColor(index, 'yellow')
            else:
                self.__setSourceBackgroundColor(index, 'transparent')


    def __setSourceBackgroundColor(self, index, color):
        if index == 0:
            self.source1.setStyleSheet('background-color: %s' % color)
        elif index == 1:
            self.source2.setStyleSheet('background-color: %s' % color)
        elif index == 2:
            self.source3.setStyleSheet('background-color: %s' % color)
        elif index == 3:
            self.source4.setStyleSheet('background-color: %s' % color)


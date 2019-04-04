from enum import Enum, unique
from math import degrees

from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
from PyQt5.QtCore import pyqtSlot

from src.app.gui.conference_ui import Ui_Conference
from src.app.controllers.conference_controller import ConferenceController

from src.app.services.videoprocessing.virtualcamera.virtual_camera_displayer import VirtualCameraDisplayer


@unique
class BtnAudioRecordLabels(Enum):
    START_RECORDING = 'Start Audio Recording'
    STOP_RECORDING = 'Stop Audio Recording'


@unique
class BtnOdasLabels(Enum):
    START_ODAS = 'Start ODAS'
    STOP_ODAS = 'Stop ODAS'


@unique
class BtnVideoLabels(Enum):
    START_VIDEO = 'Start Video'
    STOP_VIDEO = 'Stop Video'


class Conference(QWidget, Ui_Conference):

    def __init__(self, parent=None):
        super(Conference, self).__init__(parent)
        self.setupUi(self)
        self.conferenceController = ConferenceController()
        self.outputFolder.setText(self.window().getSetting('outputFolder'))

        self.virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)

        self.btnStartStopAudioRecord.setDisabled(True)

        # Qt signal slots
        self.btnSelectOutputFolder.clicked.connect(self.selectOutputFolder)
        self.btnStartStopOdas.clicked.connect(self.btnStartStopOdasClicked)
        self.btnStartStopVideo.clicked.connect(self.btnStartStopVideoClicked)
        self.btnStartStopAudioRecord.clicked.connect(self.btnStartStopAudioRecordClicked)

        self.conferenceController.signalAudioPositions.connect(self.positionDataReceived)
        self.conferenceController.signalOdasState.connect(self.odasStateChanged)
        self.conferenceController.signalRecordingState.connect(self.recordingStateChanged)
        self.conferenceController.signalVideoProcessorState.connect(self.videoProcessorStateChanged)
        self.conferenceController.signalVirtualCamerasReceived.connect(self.updateVirtualCamerasDispay)
        self.conferenceController.signalHumanSourcesDetected.connect(self.showHumanSources)
        self.conferenceController.signalException.connect(self.exceptionReceived)


    @pyqtSlot()
    def selectOutputFolder(self):
        try:
            outputFolder = QFileDialog.getExistingDirectory(
                parent=self, 
                caption='Select Output Directory', 
                directory=self.window().rootDirectory,
                options=QFileDialog.DontUseNativeDialog
            )
            if outputFolder:
                self.outputFolder.setText(outputFolder)
                self.window().setSetting('outputFolder', outputFolder)

        except Exception as e:
            self.window().emitToExceptionManager(e)


    @pyqtSlot()
    def btnStartStopOdasClicked(self):
        self.btnStartStopOdas.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopOdas.text() == BtnOdasLabels.START_ODAS.value:
            self.conferenceController.startOdasLive(odasPath=self.window().getSetting('odasPath'), micConfigPath=self.window().getSetting('micConfigPath'))
        else:
            self.conferenceController.stopOdasLive()


    @pyqtSlot()
    def btnStartStopVideoClicked(self):
        self.btnStartStopVideo.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopVideo.text() == BtnVideoLabels.START_VIDEO.value:
            self.conferenceController.startVideoProcessor(self.window().getSetting('cameraConfigPath'), self.window().getSetting('faceDetection'))
        else:
            self.conferenceController.stopVideoProcessor()


    @pyqtSlot()
    def btnStartStopAudioRecordClicked(self):
        if not self.outputFolder.text():
            self.window().emitToExceptionsManager(Exception('output folder cannot be empty'))

        self.btnStartStopAudioRecord.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopAudioRecord.text() == BtnAudioRecordLabels.START_RECORDING.value:
            self.conferenceController.startAudioRecording(self.outputFolder.text())

        else:
            self.conferenceController.saveAudioRecording()


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
    def recordingStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopAudioRecord.setText(BtnAudioRecordLabels.STOP_RECORDING.value)
        else:
            self.btnStartStopAudioRecord.setText(BtnAudioRecordLabels.START_RECORDING.value)

        self.btnStartStopAudioRecord.setDisabled(False)


    @pyqtSlot(bool)
    def odasStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopOdas.setText(BtnOdasLabels.STOP_ODAS.value)
            self.btnStartStopAudioRecord.setDisabled(False)
        else:
            self.btnStartStopOdas.setText(BtnOdasLabels.START_ODAS.value)
            self.btnStartStopAudioRecord.setText(BtnAudioRecordLabels.START_RECORDING.value)
            self.conferenceController.saveAudioRecording()
            self.btnStartStopAudioRecord.setDisabled(True)

        self.btnStartStopOdas.setDisabled(False)

    
    @pyqtSlot(bool)
    def videoProcessorStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopVideo.setText(BtnVideoLabels.STOP_VIDEO.value)
            self.virtualCameraDisplayer.startDisplaying()
            
        else:
            self.btnStartStopVideo.setText(BtnVideoLabels.START_VIDEO.value)
            self.virtualCameraDisplayer.stopDisplaying()

        self.btnStartStopVideo.setDisabled(False)


    @pyqtSlot(Exception)
    def exceptionReceived(self, e):
        self.window().emitToExceptionsManager(e)


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.conferenceController.stopVideoProcessor()
            self.conferenceController.stopAudioRecording()
            self.conferenceController.stopOdasServer()
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


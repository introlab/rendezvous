from enum import Enum, unique
from math import degrees

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget

from src.app.application_container import ApplicationContainer
from src.app.controllers.recording_controller import RecordingController
from src.app.gui.recording_ui import Ui_Recording
from src.app.services.videoprocessing.virtualcamera.virtual_camera_displayer import VirtualCameraDisplayer


@unique
class BtnRecordLabels(Enum):
    START_RECORDING = 'Start Recording'
    STOP_RECORDING = 'Stop Recording'


class Recording(QWidget, Ui_Recording):

    def __init__(self, parent=None):
        super(Recording, self).__init__(parent)
        self.setupUi(self)
        self.recordingController = RecordingController(ApplicationContainer.settings().getValue('outputFolder'))

        self.virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)

        # Qt signal slots
        self.btnStartStopRecord.clicked.connect(self.btnStartStopRecordClicked)

        self.recordingController.signalOdasState.connect(self.odasStateChanged)
        self.recordingController.signalRecordingState.connect(self.recordingStateChanged)
        self.recordingController.signalVideoProcessorState.connect(self.videoProcessorStateChanged)
        self.recordingController.signalVirtualCamerasReceived.connect(self.updateVirtualCamerasDispay)
        self.recordingController.signalException.connect(self.exceptionReceived)
        self.recordingController.transcriptionReady.connect(self.onTranscriptionReady)


    @pyqtSlot()
    def btnStartStopRecordClicked(self):
        self.btnStartStopRecord.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopRecord.text() == BtnRecordLabels.START_RECORDING.value:
            self.recordingController.startRecording()
            self.recordingController.startOdasLive(ApplicationContainer.settings().getValue('odasPath'), ApplicationContainer.settings().getValue('micConfigPath'))
            self.recordingController.startVideoProcessor(ApplicationContainer.settings().getValue('cameraConfigPath'), ApplicationContainer.settings().getValue('faceDetection'))

        else:
            self.recordingController.saveRecording()
            self.recordingController.stopOdasLive()
            self.recordingController.stopVideoProcessor()


    @pyqtSlot(object)        
    def updateVirtualCamerasDispay(self, virtualCameraImages):
        self.virtualCameraDisplayer.updateDisplay(virtualCameraImages)


    @pyqtSlot(bool)
    def recordingStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopRecord.setText(BtnRecordLabels.STOP_RECORDING.value)

        else:
            self.btnStartStopRecord.setText(BtnRecordLabels.START_RECORDING.value)

        self.btnStartStopRecord.setDisabled(False)


    @pyqtSlot(bool)
    def odasStateChanged(self, isRunning):
        if self.recordingController and not self.recordingController.videoProcessorState:
            self.btnStartStopRecord.setDisabled(False)

    
    @pyqtSlot(bool)
    def videoProcessorStateChanged(self, isRunning):
        if isRunning:
            self.virtualCameraDisplayer.startDisplaying()
        else:
            self.virtualCameraDisplayer.stopDisplaying()

        if self.recordingController and not self.recordingController.isOdasLiveConnected:
            self.btnStartStopRecord.setDisabled(False)


    @pyqtSlot(Exception)
    def exceptionReceived(self, e):
        ApplicationContainer.exceptions().show(e)


    @pyqtSlot()
    def onTranscriptionReady(self):
        ApplicationContainer.informations().show('Transcription is done')


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.recordingController.stopOdasServer()
            self.recordingController.stopVideoProcessor()
            self.recordingController.stopRecording()
            self.recordingController.cancelTranscription()
            event.accept()


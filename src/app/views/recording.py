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
        self.outputFolder.setText(ApplicationContainer.settings().getValue('outputFolder'))
        self.recordingController = RecordingController(self.outputFolder.text())

        self.virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)

        # Qt signal slots
        self.btnSelectOutputFolder.clicked.connect(self.selectOutputFolder)
        self.btnStartStopRecord.clicked.connect(self.btnStartStopRecordClicked)

        self.recordingController.signalOdasState.connect(self.odasStateChanged)
        self.recordingController.signalRecordingState.connect(self.recordingStateChanged)
        self.recordingController.signalVideoProcessorState.connect(self.videoProcessorStateChanged)
        self.recordingController.signalVirtualCamerasReceived.connect(self.updateVirtualCamerasDispay)
        self.recordingController.signalException.connect(self.exceptionReceived)


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
                ApplicationContainer.settings().setValue('outputFolder', outputFolder)

        except Exception as e:
            self.window().emitToExceptionManager(e)


    @pyqtSlot()
    def btnStartStopRecordClicked(self):
        if not self.outputFolder.text():
            ApplicationContainer.exceptions().show(Exception('output folder cannot be empty'))

        self.btnStartStopRecord.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopRecord.text() == BtnRecordLabels.START_RECORDING.value:
            self.recordingController.startRecording(self.outputFolder.text())
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


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.recordingController.stopOdasServer()
            self.recordingController.stopVideoProcessor()
            self.recordingController.stopRecording()
            event.accept()


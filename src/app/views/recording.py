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

        self.__recordingController = RecordingController()
        self.__recordingController.signalRecordingState.connect(self.__recordingStateChanged)
        self.__recordingController.signalVirtualCamerasReceived.connect(self.__updateVirtualCamerasDispay)
        self.__recordingController.signalException.connect(self.__exceptionReceived)
        self.recordingController.transcriptionReady.connect(self.onTranscriptionReady)
        
        self.__virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)

        # Qt signal slots
        self.btnStartStopRecord.clicked.connect(self.__btnStartStopRecordClicked)


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.__recordingController.close()
            event.accept()


    @pyqtSlot()
    def __btnStartStopRecordClicked(self):
        self.btnStartStopRecord.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopRecord.text() == BtnRecordLabels.START_RECORDING.value:
            self.__recordingController.startRecording(ApplicationContainer.settings().getValue('outputFolder'),
                                                      ApplicationContainer.settings().getValue('odasPath'),
                                                      ApplicationContainer.settings().getValue('micConfigPath'),
                                                      ApplicationContainer.settings().getValue('cameraConfigPath'),
                                                      ApplicationContainer.settings().getValue('faceDetection'))
        else:
            self.__recordingController.stopRecording()


    @pyqtSlot(object)        
    def __updateVirtualCamerasDispay(self, virtualCameraImages):
        self.__virtualCameraDisplayer.updateDisplay(virtualCameraImages)


    @pyqtSlot(bool)
    def __recordingStateChanged(self, isRunning):        
        if isRunning:
            self.btnStartStopRecord.setText(BtnRecordLabels.STOP_RECORDING.value)
            self.__virtualCameraDisplayer.startDisplaying()
        else:
            self.btnStartStopRecord.setText(BtnRecordLabels.START_RECORDING.value)
            self.__virtualCameraDisplayer.stopDisplaying()

        self.btnStartStopRecord.setDisabled(False)


    @pyqtSlot(Exception)
    def __exceptionReceived(self, e):
        ApplicationContainer.exceptions().show(e)

from enum import Enum, unique

from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
from PyQt5.QtCore import pyqtSlot

from src.app.gui.conference_ui import Ui_Conference
from src.app.controllers.conference_controller import ConferenceController

from src.app.services.videoprocessing.virtualcamera.virtual_camera_displayer import VirtualCameraDisplayer


@unique
class BtnRecordLabels(Enum):
    START_RECORDING = 'Start Recording'
    STOP_RECORDING = 'Stop Recording'


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
        self.outputFolder.setText(self.window().getSetting('outputFolder'))
        self.conferenceController = ConferenceController(self.outputFolder.text())

        self.virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)

        # Qt signal slots
        self.btnSelectOutputFolder.clicked.connect(self.selectOutputFolder)
        self.btnStartStopOdas.clicked.connect(self.btnStartStopOdasClicked)
        self.btnStartStopVideo.clicked.connect(self.btnStartStopVideoClicked)
        self.btnStartStopRecord.clicked.connect(self.btnStartStopRecordClicked)

        self.conferenceController.signalAudioPositions.connect(self.positionDataReceived)
        self.conferenceController.signalOdasState.connect(self.odasStateChanged)
        self.conferenceController.signalRecordingState.connect(self.recordingStateChanged)
        self.conferenceController.signalVideoProcessorState.connect(self.videoProcessorStateChanged)
        self.conferenceController.signalVirtualCamerasReceived.connect(self.updateVirtualCamerasDispay)
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
            self.btnStartStopRecord.setDisabled(True)
            self.conferenceController.startOdasLive(odasPath=self.window().getSetting('odasPath'), micConfigPath=self.window().getSetting('micConfigPath'))
        else:
            self.conferenceController.stopOdasLive()
            self.btnStartStopRecord.setDisabled(False)


    @pyqtSlot()
    def btnStartStopVideoClicked(self):
        self.btnStartStopVideo.setDisabled(True)
        self.btnStartStopRecord.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopVideo.text() == BtnVideoLabels.START_VIDEO.value:
            self.btnStartStopRecord.setDisabled(True)
            self.conferenceController.startVideoProcessor(self.window().getSetting('cameraConfigPath'), self.window().getSetting('faceDetection'))
        else:
            self.conferenceController.stopVideoProcessor()
            self.btnStartStopRecord.setDisabled(False)


    @pyqtSlot()
    def btnStartStopRecordClicked(self):
        if not self.outputFolder.text():
            self.window().emitToExceptionsManager(Exception('output folder cannot be empty'))

        self.btnStartStopRecord.setDisabled(True)
        self.btnStartStopOdas.setDisabled(True)
        self.btnStartStopVideo.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopRecord.text() == BtnRecordLabels.START_RECORDING.value:
            self.conferenceController.startOdasLive(self.window().getSetting('odasPath'), self.window().getSetting('micConfigPath'))
            self.startVideoProcessor()
            self.conferenceController.startRecording(self.outputFolder.text())

        else:
            self.conferenceController.stopOdasLive()
            self.stopVideoProcessor()
            self.conferenceController.saveRecording()


    @pyqtSlot(object)
    def positionDataReceived(self, values):
        self.source1AzimuthValueLabel.setText('%.5f' % values[0]['azimuth'])
        self.source2AzimuthValueLabel.setText('%.5f' % values[1]['azimuth'])
        self.source3AzimuthValueLabel.setText('%.5f' % values[2]['azimuth'])
        self.source4AzimuthValueLabel.setText('%.5f' % values[3]['azimuth'])

        self.source1ElevationValueLabel.setText('%.5f' % values[0]['elevation'])
        self.source2ElevationValueLabel.setText('%.5f' % values[1]['elevation'])
        self.source3ElevationValueLabel.setText('%.5f' % values[2]['elevation'])
        self.source4ElevationValueLabel.setText('%.5f' % values[3]['elevation'])


    @pyqtSlot(object, object)
    def updateVirtualCamerasDispay(self, image, virtualCameras):
        self.virtualCameraDisplayer.updateDisplay(image, virtualCameras)


    @pyqtSlot(bool)
    def recordingStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopRecord.setText(BtnRecordLabels.STOP_RECORDING.value)
        else:
            self.btnStartStopRecord.setText(BtnRecordLabels.START_RECORDING.value)
            self.btnStartStopOdas.setDisabled(False)
            self.btnStartStopVideo.setDisabled(False)

        self.btnStartStopRecord.setDisabled(False)


    @pyqtSlot(bool)
    def odasStateChanged(self, isRunning):
        if isRunning:
            if not self.conferenceController.isRecording:
                self.btnStartStopOdas.setText(BtnOdasLabels.STOP_ODAS.value)
            if self.conferenceController.isRecording:
                self.btnStartStopRecord.setDisabled(False)
        else:
            if not self.conferenceController.isRecording:
                self.btnStartStopOdas.setText(BtnOdasLabels.START_ODAS.value)
            self.btnStartStopRecord.setText(BtnRecordLabels.START_RECORDING.value)
            self.conferenceController.saveRecording()
            if not self.videoProcessor.isRunning:
                self.btnStartStopRecord.setDisabled(False)

        self.btnStartStopOdas.setDisabled(False)
        if self.conferenceController.isRecording:
            self.btnStartStopOdas.setDisabled(True)

    
    @pyqtSlot(bool)
    def videoProcessorStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopVideo.setText(BtnVideoLabels.STOP_VIDEO.value)
        else:
            self.btnStartStopVideo.setText(BtnVideoLabels.START_VIDEO.value)


    @pyqtSlot(Exception)
    def exceptionReceived(self, e):
        self.window().emitToExceptionsManager(e)


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.conferenceController.stopVideoProcessor()
            self.conferenceController.stopRecording()
            self.conferenceController.stopOdasServer()
            event.accept()


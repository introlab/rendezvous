import math
import copy
from enum import Enum, unique

from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
from PyQt5.QtCore import pyqtSlot

from src.app.gui.conference_ui import Ui_Conference
from src.app.controllers.conference_controller import ConferenceController

from src.app.services.videoprocessing.video_processor import VideoProcessor
from src.app.services.videoprocessing.virtualcamera.virtual_camera_displayer import VirtualCameraDisplayer
from src.app.services.sourceclassifier.source_classifier import SourceClassifier
from src.utils.spherical_angles_converter import SphericalAnglesConverter


@unique
class BtnAudioRecordLabels(Enum):
    START_RECORDING = 'Start Audio Recording'
    STOP_RECORDING = 'Stop Audio Recording'


@unique
class BtnOdasLabels(Enum):
    START_ODAS = 'Start ODAS'
    STOP_ODAS = 'Stop ODAS'


class Conference(QWidget, Ui_Conference):

    def __init__(self, parent=None):
        super(Conference, self).__init__(parent)
        self.setupUi(self)
        self.conferenceController = ConferenceController()
        self.outputFolder.setText(self.window().getSetting('outputFolder'))

        self.videoProcessor = VideoProcessor()
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
        self.conferenceController.signalException.connect(self.exceptionReceived)
        
        self.videoProcessor.signalException.connect(self.videoExceptionHandling)

        self.soundSources = {}


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

        if not self.videoProcessor.isRunning:
            self.btnStartStopVideo.setText('Stop Video')
            self.startVideoProcessor()
        else:
            self.stopVideoProcessor()
            self.btnStartStopVideo.setText('Start Video')
        
        self.btnStartStopVideo.setDisabled(False)


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
        # Convert angles in degrees for display
        valuesInDeg = copy.deepcopy(values)
        for source in valuesInDeg:
            source['azimuth'] = math.degrees(source['azimuth'])
            source['elevation'] = math.degrees(source['elevation'])

        self.source1AzimuthValueLabel.setText('%.5f' % valuesInDeg[0]['azimuth'])
        self.source2AzimuthValueLabel.setText('%.5f' % valuesInDeg[1]['azimuth'])
        self.source3AzimuthValueLabel.setText('%.5f' % valuesInDeg[2]['azimuth'])
        self.source4AzimuthValueLabel.setText('%.5f' % valuesInDeg[3]['azimuth'])

        self.source1ElevationValueLabel.setText('%.5f' % valuesInDeg[0]['elevation'])
        self.source2ElevationValueLabel.setText('%.5f' % valuesInDeg[1]['elevation'])
        self.source3ElevationValueLabel.setText('%.5f' % valuesInDeg[2]['elevation'])
        self.source4ElevationValueLabel.setText('%.5f' % valuesInDeg[3]['elevation'])

        self.soundSources = values


    @pyqtSlot(object, object)
    def imageReceived(self, image, virtualCameras):
        if(self.soundSources):
            cameraParams = self.videoProcessor.getCameraParams()
            rangeThreshold = 15
            sourceClassifier = SourceClassifier(cameraParams, rangeThreshold)
            sourceClassifier.classifySources(virtualCameras, self.soundSources)
            print('human sources = ' + str(sourceClassifier.getHumanSources()))
        
        self.virtualCameraDisplayer.updateDisplay(image, virtualCameras)


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


    @pyqtSlot(Exception)
    def exceptionReceived(self, e):
        self.window().emitToExceptionsManager(e)


    @pyqtSlot(Exception)
    def videoExceptionHandling(self, e):
        self.window().emitToExceptionsManager(e)

        # We make sure the thread is stopped
        self.stopVideoProcessor()

        self.btnStartStopVideo.setText('Start Video')      
        self.btnStartStopVideo.setDisabled(False)


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.stopVideoProcessor()
            self.conferenceController.stopAudioRecording()
            self.conferenceController.stopOdasServer()
            event.accept()


    def startVideoProcessor(self):
        if self.videoProcessor and not self.videoProcessor.isRunning:
            self.videoProcessor.signalFrameData.connect(self.imageReceived)
            self.videoProcessor.start(self.window().getSetting('cameraConfigPath'))


    def stopVideoProcessor(self):
        if self.videoProcessor and self.videoProcessor.isRunning:
            self.videoProcessor.stop()
            self.videoProcessor.signalFrameData.disconnect(self.imageReceived)
            self.virtualCameraDisplayer.clear()


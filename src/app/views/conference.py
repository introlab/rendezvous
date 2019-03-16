from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
from PyQt5.QtCore import pyqtSlot

from src.app.gui.conference_ui import Ui_Conference
from src.app.controllers.conference_controller import ConferenceController

from src.app.services.videoprocessing.video_processor import VideoProcessor
from src.app.services.virtualcamera.virtual_camera_displayer import VirtualCameraDisplayer
from src.app.services.virtualcamera.virtual_camera_manager import VirtualCameraManager


class Conference(QWidget, Ui_Conference):

    def __init__(self, parent=None):
        super(Conference, self).__init__(parent)
        self.setupUi(self)
        self.outputFolder.setText(self.window().getSetting('outputFolder'))

        self.videoProcessor = VideoProcessor()
        self.virtualCameraManager = VirtualCameraManager()
        self.virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)

        self.btnStartStopAudioRecord.setDisabled(True)

        # Qt signal slots
        self.btnSelectOutputFolder.clicked.connect(self.selectOutputFolder)
        self.btnStartStopOdas.clicked.connect(self.btnStartStopOdasClicked)
        self.btnStartStopVideo.clicked.connect(self.btnStartStopVideoClicked)
        self.btnStartStopAudioRecord.clicked.connect(self.btnStartStopAudioRecordClicked)

        self.videoProcessor.signalException.connect(self.videoExceptionHandling)


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

        if not self.odasStream.isRunning:
            self.startOdas()
            self.btnStartStopOdas.setText('Stop ODAS')

        else:
            self.stopOdas()
            self.btnStartStopOdas.setText('Start ODAS')
            self.audioStream.closeConnection()
            self.stopAudioRecording()

        self.btnStartStopOdas.setDisabled(False)


    @pyqtSlot()
    def btnStartStopVideoClicked(self):
        self.btnStartStopVideo.setDisabled(True)
        QApplication.processEvents()

        if not self.videoProcessor.isRunning:
            self.btnStartStopVideo.setText('Stop Video')
            self.startVideoProcessor()
        else:
            self.stopVideoProcessor()
            self.btnStartStopVideo.setText('Start Video')
        
        self.btnStartStopVideo.setDisabled(False)


    @pyqtSlot()
    def btnStartStopAudioRecordClicked(self):
        self.btnStartStopAudioRecord.setDisabled(True)
        QApplication.processEvents()

        if not self.isRecording:
            self.btnStartStopAudioRecord.setText('Stop Audio Recording')
            self.startAudioRecording()
        else:
            self.stopAudioRecording()
            self.btnStartStopAudioRecord.setText('Start Audio Recording')

        self.btnStartStopAudioRecord.setDisabled(False)


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
    def imageReceived(self, image, faces):
        imageHeight, imageWidth, _ = image.shape

        for face in faces:
            self.virtualCameraManager.addOrUpdateVirtualCamera(face, imageWidth, imageHeight)

        self.virtualCameraDisplayer.updateDisplay(image, self.virtualCameraManager.virtualCameras)

    
    @pyqtSlot()
    def audioStreamStarted(self):
        self.btnStartStopOdas.setText('Start ODAS')
        self.btnStartStopOdas.setDisabled(False)
        self.btnStartStopAudioRecord.setDisabled(True)
        self.btnStartStopAudioRecord.setText('Start Audio Recording')


    @pyqtSlot()
    def audioStreamStopped(self):
        self.stopOdas()
        self.btnStartStopOdas.setText('Start ODAS')
        self.btnStartStopOdas.setDisabled(True)
        self.btnStartStopAudioRecord.setText('Start Audio Recording')
        self.btnStartStopAudioRecord.setDisabled(True)


    @pyqtSlot()
    def recordingStarted(self):
        self.btnStartStopAudioRecord.setText('Stop Audio Recording')


    @pyqtSlot()
    def recordingStopped(self):
        self.btnStartStopAudioRecord.setText('Start Audio Recording')


    @pyqtSlot()
    def odasStarted(self):
        self.btnStartStopAudioRecord.setDisabled(False)


    @pyqtSlot()
    def odasStopped(self):
        self.btnStartStopAudioRecord.setDisabled(True)

    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.stopVideoProcessor()
            self.stopOdas()
            self.audioStream.stop()
            self.audioWriter.stop()
            event.accept()


    def startVideoProcessor(self):
        if self.videoProcessor and not self.videoProcessor.isRunning:
            self.videoProcessor.signalFrameData.connect(self.imageReceived)
            self.videoProcessor.start(debug=False, 
                                      cameraConfigPath=self.window().getSetting('cameraConfigPath'))


    def stopVideoProcessor(self):
        if self.videoProcessor and self.videoProcessor.isRunning:
            self.videoProcessor.stop()
            self.videoProcessor.signalFrameData.disconnect(self.imageReceived)
            self.virtualCameraManager.virtualCameras.clear()


    @pyqtSlot(Exception)
    def videoExceptionHandling(self, e):
        self.window().emitToExceptionManager(e)

        # We make sure the thread is stopped
        self.stopVideoProcessor()

        self.btnStartStopVideo.setText('Start Video')      
        self.btnStartStopVideo.setDisabled(False)


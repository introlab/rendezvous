from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot

from src.app.gui.odas_live_ui import Ui_OdasLive
from src.app.odasstream.odas_stream import OdasStream
from src.app.videoprocessing.video_processor import VideoProcessor
from src.app.virtualcamera.virtual_camera_displayer import VirtualCameraDisplayer
from src.app.virtualcamera.virtual_camera_manager import VirtualCameraManager


class OdasLive(QWidget, Ui_OdasLive):

    def __init__(self, parent=None):
        super(OdasLive, self).__init__(parent)
        self.setupUi(self)
        self.odasStream = OdasStream()
        self.videoProcessor = VideoProcessor()

        self.virtualCameraManager = VirtualCameraManager()
        self.virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)

        self.btnStartStopAudioRecord.setDisabled(True)

        # Qt signal slots
        self.btnSelectOutputFolder.clicked.connect(self.selectOutputFolder)
        self.btnStartStopOdas.clicked.connect(self.btnStartStopOdasClicked)
        self.btnStartStopVideo.clicked.connect(self.btnStartStopVideoClicked)
        self.btnStartStopAudioRecord.clicked.connect(self.btnStartStopAudioRecordClicked)

        self.odasStream.signalOdasException.connect(self.odasExceptionHandling)
        self.videoProcessor.signalVideoException.connect(self.videoExceptionHandling)
    

    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.stopVideoProcessor()
            self.stopOdas()
            event.accept()


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
        except Exception as e:
            self.window().emitToExceptionManager(e)


    def startOdas(self):
        if self.odasStream and not self.odasStream.isRunning:
            self.odasStream.signalOdasData.connect(self.odasDataReveived)
            self.odasStream.start(odasPath=self.window().settingsManager.getValue('odasPath'), 
                                  micConfigPath=self.window().settingsManager.getValue('micConfigPath'))


    def stopOdas(self):
        if self.odasStream and self.odasStream.isRunning:
            self.odasStream.signalOdasData.disconnect(self.odasDataReveived)
            self.odasStream.stop()

    
    def startAudioRecording(self):
        pass


    def stopAudioRecording(self):
        pass


    def startVideoProcessor(self):
        if self.videoProcessor and not self.videoProcessor.isRunning:
            self.videoProcessor.signalFrameData.connect(self.imageReceived)
            self.videoProcessor.start(debug=False, 
                                      cameraConfigPath=self.window().settingsManager.getValue('cameraConfigPath'))


    def stopVideoProcessor(self):
        if self.videoProcessor and self.videoProcessor.isRunning:
            self.videoProcessor.stop()
            self.videoProcessor.signalFrameData.disconnect(self.imageReceived)
            self.virtualCameraManager.virtualCameras.clear()


    def odasExceptionHandling(self, e):
        self.window().exceptionManager.signalException.emit(e)

        # We make sure the thread is stopped
        self.stopOdas()

        self.btnStartStopOdas.setText('Start ODAS')
        self.btnStartStopOdas.setDisabled(False)
        self.btnStartStopAudioRecord.setDisabled(True)


    def videoExceptionHandling(self, e):
        self.window().exceptionManager.signalException.emit(e)

        # We make sure the thread is stopped
        self.stopVideoProcessor()

        self.btnStartStopVideo.setText('Start Video')      
        self.btnStartStopVideo.setDisabled(False)


    @pyqtSlot()
    def btnStartStopOdasClicked(self):
        self.btnStartStopOdas.setDisabled(True)

        if not self.odasStream.isRunning:
            self.startOdas()
            self.btnStartStopOdas.setText('Stop ODAS')
            self.btnStartStopAudioRecord.setDisabled(False)

        else:
            self.stopOdas()
            self.btnStartStopOdas.setText('Start ODAS')
            self.stopAudioRecording()
            self.btnStartStopAudioRecord.setDisabled(True)

        self.btnStartStopOdas.setDisabled(False)


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
        self.btnStartStopAudioRecord.setDisabled(True)

        self.btnStartStopAudioRecord.setDisabled(False)


    @pyqtSlot(object)
    def odasDataReveived(self, values):
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
        imageHeight, imageWidth, colors = image.shape

        for face in faces:
            self.virtualCameraManager.addOrUpdateVirtualCamera(face, imageWidth, imageHeight)

        self.virtualCameraDisplayer.updateDisplay(image, self.virtualCameraManager.virtualCameras)

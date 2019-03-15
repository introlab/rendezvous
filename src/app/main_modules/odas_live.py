from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot

from src.app.gui.odas_live_ui import Ui_OdasLive
from src.app.odasstream.odas_stream import OdasStream
from src.app.videoprocessing.video_processor import VideoProcessor
from src.app.recorder.audio.audio_stream import AudioStream
from src.app.recorder.audio.audio_writer import AudioWriter
from src.app.virtualcamera.virtual_camera_displayer import VirtualCameraDisplayer
from src.app.virtualcamera.virtual_camera_manager import VirtualCameraManager


class OdasLive(QWidget, Ui_OdasLive):

    def __init__(self, parent=None):
        super(OdasLive, self).__init__(parent)
        self.setupUi(self)
        self.odasStream = OdasStream()
        self.audioStream = AudioStream('127.0.0.1', 10020)
        self.startAudioStreaming()
        self.audioWriter = AudioWriter()

        self.videoProcessor = VideoProcessor()
        self.virtualCameraManager = VirtualCameraManager()
        self.virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)

        self.btnStartStopAudioRecord.setDisabled(True)
        self.outputFolder.setText(self.window().settingsManager.getValue('outputFolder'))

        # Qt signal slots
        self.btnSelectOutputFolder.clicked.connect(self.selectOutputFolder)
        self.btnStartStopOdas.clicked.connect(self.btnStartStopOdasClicked)
        self.btnStartStopVideo.clicked.connect(self.btnStartStopVideoClicked)
        self.btnStartStopAudioRecord.clicked.connect(self.btnStartStopAudioRecordClicked)

        self.odasStream.signalException.connect(self.odasExceptionHandling)
        self.videoProcessor.signalException.connect(self.videoExceptionHandling)
        self.audioStream.signalException.connect(self.audioStreamExceptionHandling)
        
        self.audioStream.signalServerUp.connect(self.audioStreamStarted)
        self.audioStream.signalServerDown.connect(self.audioStreamStopped)
        
        self.audioWriter.signalRecordingStarted.connect(self.recordingStarted)
        self.audioWriter.signalRecordingStopped.connect(self.recordingStopped)


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
                self.window().settingsManager.setValue('outputFolder', outputFolder)

        except Exception as e:
            self.window().emitToExceptionManager(e)


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
            self.audioStream.closeConnection(isFromUI=True)
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

        if not self.audioWriter.isRecording:
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

    
    @pyqtSlot(bytes)
    def audioDataReceived(self, streamData):
        if self.audioWriter and self.audioWriter.mailbox:
            self.audioWriter.mailbox.put(streamData)

    
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


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.stopVideoProcessor()
            self.audioStream.stopServer()
            self.stopAudioRecording()
            self.stopOdas()
            event.accept()


    def startOdas(self):
        if self.odasStream and not self.odasStream.isRunning:
            self.odasStream.signalOdasData.connect(self.positionDataReceived)
            self.odasStream.start(odasPath=self.window().settingsManager.getValue('odasPath'), 
                                  micConfigPath=self.window().settingsManager.getValue('micConfigPath'))


    def stopOdas(self):
        if self.odasStream and self.odasStream.isRunning:
            self.odasStream.signalOdasData.disconnect(self.positionDataReceived)
            self.odasStream.stop()


    def startAudioStreaming(self):
        if self.audioStream and not self.audioStream.isRunning:
            self.audioStream.startServer(isVerbose=True)


    def stopAudioStreaming(self):
        if self.audioStream and self.audioStream.isRunning:
            self.audioStream.stopServer()


    def startAudioRecording(self):
        try:
            if not self.audioWriter.isRecording:
                self.audioWriter.changeWavSettings(outputFolder=self.outputFolder.text(), nChannels=1, byteDepth=2, sampleRate=48000)
                self.audioWriter.start()
                self.audioStream.signalNewData.connect(self.audioDataReceived)

        except Exception as e:
            self.window().exceptionManager.signalException.emit(e)


    def stopAudioRecording(self):
        try:
            if self.audioWriter.isRecording:
                # stop data reception for audiowriter and stop recording.
                self.audioStream.signalNewData.disconnect(self.audioDataReceived)
                self.audioWriter.stop()

        except Exception as e:
            self.window().exceptionManager.signalException.emit(e)


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

        # We make sure the threads are stopped
        self.audioStream.closeConnection()
        self.stopOdas()
        self.stopAudioRecording()

        self.btnStartStopOdas.setText('Start ODAS')
        self.btnStartStopOdas.setDisabled(False)
        self.btnStartStopAudioRecord.setDisabled(True)


    def audioStreamExceptionHandling(self, e):
        self.stopAudioStreaming()
        self.stopAudioRecording()
        self.btnStartStopAudioRecord.setDisabled(True)
        self.btnStartStopOdas.setDisabled(True)
        self.window().exceptionManager.signalException.emit(e)


    def videoExceptionHandling(self, e):
        self.window().exceptionManager.signalException.emit(e)

        # We make sure the thread is stopped
        self.stopVideoProcessor()

        self.btnStartStopVideo.setText('Start Video')      
        self.btnStartStopVideo.setDisabled(False)


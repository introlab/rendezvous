from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.app.services.recorder.recorder import Recorder, RecorderActions
from src.app.services.videoprocessing.video_processor import VideoProcessor


class RecordingController(QObject):

    signalException = pyqtSignal(Exception)
    signalOdasState = pyqtSignal(bool)
    signalRecordingState = pyqtSignal(bool)
    signalVideoProcessorState = pyqtSignal(bool)
    signalVirtualCamerasReceived = pyqtSignal(object, object)

    def __init__(self, outputFolder, odasserver, parent=None):
        super(RecordingController, self).__init__(parent)
        self.__odasserver = odasserver

        self.__recorder = Recorder(outputFolder)
        self.__recorder.changeAudioSettings(outputFolder=outputFolder, nChannels=4, nChannelFile=1, byteDepth=2, sampleRate=48000)
        self.__recorder.start()

        self.isRecording = False
        self.isOdasLiveConnected = False
        self.videoProcessorState = False

        self.__recorder.signalException.connect(self.exceptionHandling)

        self.__odasserver.signalException.connect(self.exceptionHandling)
        self.__odasserver.signalAudioData.connect(self.audioDataReceived)
        self.__odasserver.signalClientsConnected.connect(self.odasClientConnected)

        self.__videoProcessor = VideoProcessor()
        self.__videoProcessor.signalException.connect(self.exceptionHandling)
        self.__videoProcessor.signalFrameData.connect(self.virtualCamerasReceived)
        self.__videoProcessor.signalStateChanged.connect(self.videoProcessorStateChanged)



    @pyqtSlot(bool)
    def odasClientConnected(self, isConnected):
        self.isOdasLiveConnected = isConnected
        self.signalOdasState.emit(isConnected)


    @pyqtSlot(bool)
    def videoProcessorStateChanged(self, state):
        self.videoProcessorState = state
        self.signalVideoProcessorState.emit(state)


    @pyqtSlot(bytes)
    def audioDataReceived(self, streamData):
        if self.isRecording and self.__recorder and self.__recorder.mailbox:
            self.__recorder.mailbox.put(('audio', streamData))


    @pyqtSlot(object, object)
    def virtualCamerasReceived(self, image, virtualCameras):
        self.signalVirtualCamerasReceived.emit(image, virtualCameras)

        if self.isRecording and self.__recorder and self.__recorder.mailbox:
            self.__recorder.mailbox.put(('video', '0', image))


    @pyqtSlot(Exception)
    def exceptionHandling(self, e):
        self.stopOdasLive()
        self.stopVideoProcessor()
        self.isRecording = False
        self.signalRecordingState.emit(self.isRecording)
        self.signalException.emit(e)


    def startOdasLive(self, odasPath, micConfigPath):
        self.__odasserver.startOdasLive(odasPath, micConfigPath)


    def stopOdasLive(self):
        self.__odasserver.stopOdasLive()


    def stopOdasServer(self):
        if self.__odasserver.isRunning:
            self.__odasserver.stop()


    def startRecording(self, outputFolder):
        try:
            if not self.isRecording:
                self.__recorder.changeAudioSettings(outputFolder=outputFolder, nChannels=4, nChannelFile=1, byteDepth=2, sampleRate=48000)
                self.__recorder.setOutputFolder(folderpath=outputFolder)
                self.__recorder.mailbox.put(RecorderActions.NEW_RECORDING)
                self.isRecording = True
                self.signalRecordingState.emit(self.isRecording)

        except Exception as e:
            self.isRecording = False
            self.signalException.emit(e)
            self.signalRecordingState.emit(self.isRecording)


    def saveRecording(self):
        if self.__recorder and self.isRecording:
            # stop data reception for recorder and save wave files
            self.isRecording = False
            self.__recorder.mailbox.put(RecorderActions.SAVE_FILES)
            self.signalRecordingState.emit(self.isRecording)


    def stopRecording(self):
        if self.__recorder:
            self.isRecording = False
            self.__recorder.stop()
            self.signalRecordingState.emit(self.isRecording)


    def startVideoProcessor(self, cameraConfigPath, faceDetection):
        if self.__videoProcessor and not self.__videoProcessor.isRunning:
            self.__videoProcessor.start(cameraConfigPath, faceDetection)


    def stopVideoProcessor(self):
        if self.__videoProcessor and self.__videoProcessor.isRunning:
            self.__videoProcessor.stop()


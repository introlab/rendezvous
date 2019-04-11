from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.app.application_container import ApplicationContainer
from src.app.services.recorder.recorder import Recorder, RecorderActions


class RecordingController(QObject):

    signalException = pyqtSignal(Exception)
    signalOdasState = pyqtSignal(bool)
    signalRecordingState = pyqtSignal(bool)
    signalVideoProcessorState = pyqtSignal(bool)
    signalVirtualCamerasReceived = pyqtSignal(object)

    def __init__(self, outputFolder, parent=None):
        super(RecordingController, self).__init__(parent)

        outputFolder = ApplicationContainer.settings().getValue('defaultOutputFolder')
        self.__recorder = Recorder(outputFolder)
        self.__recorder.changeAudioSettings(outputFolder=outputFolder, nChannels=4, nChannelFile=1, byteDepth=2, sampleRate=48000)
        self.__recorder.start()

        self.isRecording = False
        self.isOdasLiveConnected = False
        self.videoProcessorState = False

        self.__recorder.signalException.connect(self.exceptionHandling)

        ApplicationContainer.odas().signalException.connect(self.exceptionHandling)
        ApplicationContainer.odas().signalAudioData.connect(self.audioDataReceived)
        ApplicationContainer.odas().signalClientsConnected.connect(self.odasClientConnected)

        ApplicationContainer.videoProcessor().signalException.connect(self.exceptionHandling)
        ApplicationContainer.videoProcessor().signalVirtualCameras.connect(self.virtualCamerasReceived)
        ApplicationContainer.videoProcessor().signalStateChanged.connect(self.videoProcessorStateChanged)


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
    def virtualCamerasReceived(self, images, virtualCameras):
        self.signalVirtualCamerasReceived.emit(images)

        if images:
            if self.isRecording and self.__recorder and self.__recorder.mailbox:
                self.__recorder.mailbox.put(('video', '0', images[0]))


    @pyqtSlot(Exception)
    def exceptionHandling(self, e):
        self.stopOdasLive()
        self.stopVideoProcessor()
        self.isRecording = False
        self.signalRecordingState.emit(self.isRecording)
        self.signalException.emit(e)


    def startOdasLive(self, odasPath, micConfigPath):
        ApplicationContainer.odas().startOdasLive(odasPath, micConfigPath)


    def stopOdasLive(self):
        ApplicationContainer.odas().stopOdasLive()


    def stopOdasServer(self):
        if ApplicationContainer.odas().isRunning:
            ApplicationContainer.odas().stop()


    def startRecording(self):
        try:
            if not self.isRecording:
                outputFolder = ApplicationContainer.settings().getValue('defaultOutputFolder')
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
        if ApplicationContainer.videoProcessor() and not ApplicationContainer.videoProcessor().isRunning:
            ApplicationContainer.videoProcessor().start(cameraConfigPath, faceDetection)


    def stopVideoProcessor(self):
        if ApplicationContainer.videoProcessor() and ApplicationContainer.videoProcessor().isRunning:
            ApplicationContainer.videoProcessor().stop()


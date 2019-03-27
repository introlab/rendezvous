from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.app.services.odas.odas import Odas
from src.app.services.recorder.recorder import Recorder, RecorderActions

class ConferenceController(QObject):

    signalException = pyqtSignal(Exception)
    signalAudioPositions = pyqtSignal(object)
    signalOdasState = pyqtSignal(bool)
    signalRecordingState = pyqtSignal(bool)

    def __init__(self, outputFolder, parent=None):
        super(ConferenceController, self).__init__(parent)

        self.__odas = Odas(hostIP='127.0.0.1', portPositions=10020, portAudio=10030, isVerbose=False)
        self.__odas.start()

        self.__recorder = Recorder(outputFolder)
        self.__recorder.changeAudioSettings(outputFolder=outputFolder, nChannels=4, nChannelFile=1, byteDepth=2, sampleRate=48000)
        self.__recorder.start()
        self.isRecording = False
        self.__odasLiveConnected = False

        self.__recorder.signalException.connect(self.recorderExceptionHandling)
        self.__odas.signalException.connect(self.odasExceptionHandling)
        

        self.__odas.signalAudioData.connect(self.audioDataReceived)
        self.__odas.signalPositionData.connect(self.positionDataReceived)
        self.__odas.signalClientsConnected.connect(self.odasClientConnected)


    @pyqtSlot(bool)
    def odasClientConnected(self, isConnected):
        self.signalOdasState.emit(isConnected)


    @pyqtSlot(bytes)
    def audioDataReceived(self, streamData):
        if self.isRecording and self.__recorder and self.__recorder.mailbox:
            self.__recorder.mailbox.put(('audio', streamData))


    @pyqtSlot(object)
    def positionDataReceived(self, positions):
        self.signalAudioPositions.emit(positions)


    @pyqtSlot(Exception)
    def odasExceptionHandling(self, e):
        self.saveRecording()
        self.isRecording = False
        self.signalRecordingState.emit(self.isRecording)
        self.signalOdasState.emit(False)
        self.signalException.emit(e)


    @pyqtSlot(Exception)
    def recorderExceptionHandling(self, e):
        self.isRecording = False
        self.signalRecordingState.emit(self.isRecording)
        self.signalException.emit(e)


    def startOdasLive(self, odasPath, micConfigPath):
        self.__odas.startOdasLive(odasPath, micConfigPath)


    def stopOdasLive(self):
        self.__odas.stopOdasLive()


    def stopOdasServer(self):
        if self.__odas.isRunning:
            self.__odas.stop()


    def startRecording(self, outputFolder):
        try:
            if not self.isRecording:
                self.__recorder.changeAudioSettings(outputFolder=outputFolder, nChannels=4, nChannelFile=1, byteDepth=2, sampleRate=48000)
                self.__recorder.changeVideoSettings(outputFolder=outputFolder)
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


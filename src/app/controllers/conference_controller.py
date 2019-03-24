from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.app.services.odas.odas import Odas
from src.app.services.recorder.audio.audio_writer import AudioWriter, WriterActions

class ConferenceController(QObject):

    signalException = pyqtSignal(Exception)
    signalAudioPositions = pyqtSignal(object)
    signalOdasState = pyqtSignal(bool)
    signalRecordingState = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(ConferenceController, self).__init__(parent)

        self.__odas = Odas(hostIP='127.0.0.1', port= 10020, isVerbose=True)
        self.__odas.start()

        self.__audioWriter = AudioWriter()
        self.__audioWriter.changeWavSettings(outputFolder='', nChannels=4, nChannelFile=1, byteDepth=2, sampleRate=48000)
        self.__audioWriter.start()
        self.__isRecording = False
        self.__odasLiveConnected = False

        self.__audioWriter.signalException.connect(self.audioWriterExceptionHandling)
        self.__odas.signalException.connect(self.odasExceptionHandling)
        

        self.__odas.signalAudioData.connect(self.audioDataReceived)
        self.__odas.signalPositionData.connect(self.positionDataReceived)
        self.__odas.signalClientsConnected.connect(self.odasClientConnected)


    @pyqtSlot(bool)
    def odasClientConnected(self, isConnected):
        self.signalOdasState.emit(isConnected)


    @pyqtSlot(bytes)
    def audioDataReceived(self, streamData):
        if self.__isRecording and self.__audioWriter and self.__audioWriter.mailbox:
            self.__audioWriter.mailbox.put(streamData)


    @pyqtSlot(object)
    def positionDataReceived(self, positions):
        self.signalAudioPositions.emit(positions)


    @pyqtSlot(Exception)
    def odasExceptionHandling(self, e):
        self.saveAudioRecording()
        self.__isRecording = False
        self.signalRecordingState.emit(self.__isRecording)
        self.signalOdasState.emit(False)
        self.signalException.emit(e)


    @pyqtSlot(Exception)
    def audioWriterExceptionHandling(self, e):
        self.__isRecording = False
        self.signalRecordingState.emit(self.__isRecording)
        self.signalException.emit(e)


    def startOdasLive(self, odasPath, micConfigPath):
        self.__odas.startOdasLive(odasPath, micConfigPath)


    def stopOdasLive(self):
        self.__odas.stopOdasLive()


    def stopOdasServer(self):
        if self.__odas.isRunning:
            self.__odas.stop()


    def startAudioRecording(self, outputFolder):
        try:
            if not self.__isRecording:
                self.__audioWriter.changeWavSettings(outputFolder=outputFolder, nChannels=4, nChannelFile=1, byteDepth=2, sampleRate=48000)
                self.__audioWriter.mailbox.put(WriterActions.NEW_RECORDING)
                self.__isRecording = True
                self.signalRecordingState.emit(self.__isRecording)

        except Exception as e:
            self.signalException.emit(e)


    def saveAudioRecording(self):
        if self.__audioWriter and self.__isRecording:
            # stop data reception for audiowriter and save wave files
            self.__isRecording = False
            self.__audioWriter.mailbox.put(WriterActions.SAVE_FILES)
            self.signalRecordingState.emit(self.__isRecording)


    def stopAudioRecording(self):
        if self.__audioWriter:
            self.__isRecording = False
            self.__audioWriter.stop()
            self.signalRecordingState.emit(self.__isRecording)


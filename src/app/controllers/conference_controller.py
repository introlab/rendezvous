from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.app.services.odas.odas import Odas
from src.app.services.recorder.audio.audio_writer import AudioWriter, WriterActions

class ConferenceController(QObject):

    signalAudioPositions = pyqtSignal(object)
    signalOdasIsRunning = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(ConferenceController, self).__init__(parent)

        self.odas = Odas(hostIP='127.0.0.1', port= 10020, isVerbose=True)
        self.odas.start()

        self.audioWriter = AudioWriter()
        self.audioWriter.changeWavSettings(outputFolder=self.outputFolder.text(), nChannels=4, nChannelFile=1, byteDepth=2, sampleRate=48000)
        self.audioWriter.start()
        self.isRecording = False
        self.odasLiveConnected = False

        self.audioWriter.signalException.connect(self.audioWriterExceptionHandling)
        self.odas.signalException.connect(self.odasExceptionHandling)
        
        self.odas.signalAudioData.connect(self.audioDataReceived)
        self.odas.signalPositionData.connect(self.positionDataReceived)
        self.odas.signalClientConnected.connect(self.odasClientConnected)


    @pyqtSlot(bool)
    def odasClientConnected(self, isConnected):
        self.signalOdasIsRunning.emit(isConnected)


    @pyqtSlot(bytes)
    def audioDataReceived(self, streamData):
        if self.isRecording and self.audioWriter and self.audioWriter.mailbox:
            self.audioWriter.mailbox.put(streamData)


    @pyqtSlot(object)
    def positionDataReceived(self, positions):
        self.signalAudioPositions.emit(positions)


    @pyqtSlot(Exception)
    def odasExceptionHandling(self, e):
        self.stopOdas()
        self.window().emitToExceptionManager(e)


    @pyqtSlot(Exception)
    def audioWriterExceptionHandling(self, e):
        self.isRecording = False
        self.window().emitToExceptionManager(e)


    def startOdasLive(self):
        self.odas.startOdasLive(odasPath=self.window().getSetting('odasPath'), micConfigPath=self.window().getSetting('micConfigPath'))


    def stopOdasLive(self):
        self.odas.stopOdasLive()


    def stopOdasServer(self):
        if self.odas.isRunning:
            self.odas.stop()


    def startAudioRecording(self):
        try:
            if not self.isRecording:
                self.audioWriter.changeWavSettings(outputFolder=self.outputFolder.text(), nChannels=4, nChannelFile=1, byteDepth=2, sampleRate=48000)
                self.audioWriter.mailbox.put(WriterActions.NEW_RECORDING)
                self.isRecording = True

        except Exception as e:
            self.window().emitToExceptionManager(e)


    def stopAudioRecording(self):
        if self.audioWriter and self.isRecording:
            # stop data reception for audiowriter and stop recording.
            self.isRecording = False
            self.audioWriter.mailbox.put(WriterActions.SAVE_FILES)



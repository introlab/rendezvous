from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from src.app.application_container import ApplicationContainer


class TranscriptionController(QObject):  

    transcriptionReady = pyqtSignal(str)
    exception = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(TranscriptionController, self).__init__(parent)

        self.speechToText = ApplicationContainer.speechToText()


    def requestTranscription(self, audioDataPath):
        if not self.speechToText.isRunning:
            # We need to set the config since we can't pass arguments directly to the thread.
            settings = ApplicationContainer.settings()
            config = {
                'audioDataPath' : audioDataPath,
                'encoding' : settings.getValue('speechToTextEncoding'),
                'enhanced' : bool(settings.getValue('speechToTextEnhanced')),
                'languageCode' : settings.getValue('speechToTextLanguage'),
                'model' : settings.getValue('speechToTextModel'),
                'outputFolder' : settings.getValue('defaultOutputFolder'),
                'sampleRate' : int(settings.getValue('speechToTextSampleRate')),
                'serviceAccountPath' : settings.getValue('serviceAccountPath')
            }
            self.speechToText.setConfig(config)
            self.connectSignals()
            self.speechToText.asynchroneSpeechToText.start()
        
        else:
            self.exception.emit(Exception('Transcription is already running'))


    def cancelTranscription(self):
        if self.speechToText.isRunning:
            self.speechToText.asynchroneSpeechToText.quit()
            self.disconnectSignals()


    def connectSignals(self):
        self.speechToText.transcriptionReady.connect(self.onTranscriptionReady)
        self.speechToText.exception.connect(self.onException)


    def disconnectSignals(self):
        self.speechToText.transcriptionReady.disconnect(self.onTranscriptionReady)
        self.speechToText.exception.disconnect(self.onException)


    @pyqtSlot(str)
    def onTranscriptionReady(self, transcription):
        self.transcriptionReady.emit(transcription)
        self.speechToText.asynchroneSpeechToText.quit()
        self.disconnectSignals()


    @pyqtSlot(Exception)
    def onException(self, e):
        self.exception.emit(e)
        self.speechToText.asynchroneSpeechToText.quit()
        self.disconnectSignals()


from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from src.app.services.speechtotext.speech_to_text import EncodingTypes, LanguageCodes, Models,SpeechToText


class TranscriptionController(QObject):  

    transcriptionReady = pyqtSignal(str)
    exception = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(TranscriptionController, self).__init__(parent)

        # Initilalization of the SpeechToText service and his worker thread.
        self.speechToText = SpeechToText()
        self.speechToTextThread = QThread()
        # Make the SpeechToText executable like a thread.
        self.speechToText.moveToThread(self.speechToTextThread)
        # What will run when the thread starts.
        self.speechToTextThread.started.connect(self.speechToText.resquestTranscription)

        # Qt signal slots.
        self.speechToText.transcriptionReady.connect(self.onTranscriptionReady)
        self.speechToText.exception.connect(self.onException)


    def getEncodingTypes(self):
        return EncodingTypes


    def getLanguageCodes(self):
        return LanguageCodes


    def getModels(self):
        return Models


    def getMinSampleRate(self):
        return self.speechToText.getMinSampleRate()


    def getMaxSampleRate(self):
        return self.speechToText.getMaxSampleRate()


    def getDefaultSampleRate(self):
        return self.speechToText.getDefaultSampleRate()


    def resquestTranscription(self, config):
        # We need to set the config since we can't pass arguments directly to the thread.
        self.speechToText.setConfig(config)
        self.speechToTextThread.start()


    @pyqtSlot(str)
    def onTranscriptionReady(self, transcription):
        self.transcriptionReady.emit(transcription)
        self.speechToTextThread.quit()


    @pyqtSlot(Exception)
    def onException(self, e):
        self.exception.emit(e)
        self.speechToTextThread.quit()


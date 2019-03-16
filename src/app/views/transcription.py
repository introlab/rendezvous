import os
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot

from src.app.controllers.speechtotext.speech_to_text import EncodingType, LanguageCode, Model, SpeechToText
from src.app.gui.transcription_ui import Ui_Transcription


class Transcription(QWidget, Ui_Transcription):

    rootDirectory = os.path.realpath(Path(__file__).parents[3])

    def __init__(self, parent=None):
        super(Transcription, self).__init__(parent)
        self.setupUi(self)

        # Populate UI.
        self.encoding.addItems([encodingType.value for encodingType in EncodingType])
        # Valid range accepted by the Google API.
        self.sampleRate.setRange(8000, 48000)
        # Value we are most likely to use.
        self.sampleRate.setValue(48000)
        self.language.addItems([languageCode.value for languageCode in LanguageCode])
        self.model.addItems([model.value for model in Model])

        # Qt signal slots.
        self.btnImportAudio.clicked.connect(self.importAudioClicked)
        self.btnTranscribe.clicked.connect(self.transcribeClicked)


    # Handles the event where the user closes the window with the X button.
    def closeEvent(self, event):
        if event:
            event.accept()


    @pyqtSlot()
    def importAudioClicked(self):
        try:
            audioDataPath, _ = QFileDialog.getOpenFileName(parent=self, 
                                                           caption='Import Audio Data', 
                                                           directory=self.rootDirectory,
                                                           options=QFileDialog.DontUseNativeDialog)
            if audioDataPath:
                self.audioDataPath.setText(audioDataPath)
        except Exception as e:
            self.window().emitToExceptionsManager(e)


    @pyqtSlot()
    def transcribeClicked(self):
        self.transcriptionResult.setText('Transcribing...')
        self.setDisabled(True)
        # The UI update is scheduled after the ending of the slot, that's why we need to force the porcessing of events.
        QApplication.processEvents()

        try:
            config = {
                'encoding' : self.encoding.currentText(),
                'sampleRate' : self.sampleRate.value(),
                'languageCode' : self.language.currentText(),
                'model' : self.model.currentText(),
                'enhanced' : self.enhanced.checkState()}

            transcriptionResult = SpeechToText.resquestTranscription(serviceAccountPath=self.window().getSetting('serviceAccountPath'),
                                                                        audioDataPath=self.audioDataPath.text(),
                                                                        config=config)
            self.transcriptionResult.setText(transcriptionResult)
        
        except Exception as e:
            self.window().emitToExceptionsManager(e)
            self.transcriptionResult.setText('')
        finally:
            self.setDisabled(False)
        
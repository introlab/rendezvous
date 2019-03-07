from enum import Enum, unique, auto
from os import path

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot

from src.app.components.dialogs.warning_box import WarningBox
from src.app.speechtotextapi.speech_to_text_api import SpeechToTextAPI
from src.app.gui.speech_to_text_ui import Ui_SpeechToText

@unique
class EncodingType(Enum):
    ENCODING_UNSPECIFIED = 'ENCODING_UNSPECIFIED'
    FLAC = 'FLAC'
    AMR = 'AMR'
    AMR_WB = 'AMR_WB'
    LINEAR16 = 'LINEAR16'
    OGG_OPUS = 'OGG_OPUS'
    SPEEX_WITH_HEADER_BYTE = 'SPEEX_WITH_HEADER_BYTE'

@unique
class LanguageCode(Enum):
    FR_CA = 'fr-CA'
    EN_CA = 'en-CA'

@unique
class Model(Enum):
    DEFAULT = 'default'
    COMMAND_AND_SEARCH = 'command_and_search'
    PHONE_CALL = 'phone_call'
    VIDEO = 'video'

class SpeechToText(QWidget, Ui_SpeechToText):
    def __init__(self, parent=None):
        super(SpeechToText, self).__init__(parent)
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
        self.btnImportAudio.clicked.connect(self.ImportAudioClicked)
        self.btnImportServiceAccount.clicked.connect(self.ImportServiceAccountClicked)
        self.btnTranscribe.clicked.connect(self.TranscribeClicked)


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        event.accept()


    @pyqtSlot()
    def ImportAudioClicked(self):
        audioDataPath = QFileDialog.getOpenFileName(self, 'Open Audio Data', './')
        self.audioDataPath.setText(audioDataPath[0])


    @pyqtSlot()
    def ImportServiceAccountClicked(self):
        serviceAccountPath = QFileDialog.getOpenFileName(self, 'Open Google Service Account File', './')
        self.serviceAccountPath.setText(serviceAccountPath[0])


    @pyqtSlot()
    def TranscribeClicked(self):
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

            self.transcriptionResult.setText(SpeechToTextAPI.resquestTranscription(
                self.serviceAccountPath.text(),
                self.audioDataPath.text(),
                config))
        
        except Exception as e:
            WarningBox(e)
            self.transcriptionResult.setText('')
        finally:
            self.setDisabled(False)
    
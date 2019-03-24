from pathlib import Path

from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot

from src.app.controllers.transcription_controller import TranscriptionController
from src.app.gui.transcription_ui import Ui_Transcription


class Transcription(QWidget, Ui_Transcription):

    rootDirectory = str(Path(__file__).resolve().parents[3])

    def __init__(self, parent=None):
        super(Transcription, self).__init__(parent)
        self.setupUi(self)

        # Initilalization of the controller.
        self.transcriptionController = TranscriptionController()



        # Populate UI.
        self.encoding.addItems([encodingType.value for encodingType in self.transcriptionController.getEncodingTypes()])
        self.sampleRate.setRange(self.transcriptionController.getMinSampleRate(), self.transcriptionController.getMaxSampleRate())
        self.sampleRate.setValue(self.transcriptionController.getDefaultSampleRate())
        self.language.addItems([languageCode.value for languageCode in self.transcriptionController.getLanguageCodes()])
        self.model.addItems([model.value for model in self.transcriptionController.getModels()])    

        # Qt signal slots.
        self.btnImportAudio.clicked.connect(self.onImportAudioClicked)
        self.btnTranscribe.clicked.connect(self.onTranscribeClicked)
        self.transcriptionController.transcriptionReady.connect(self.onTranscriptionReady)
        self.transcriptionController.exception.connect(self.onException)


    # Handles the event where the user closes the window with the X button.
    def closeEvent(self, event):
        if event:
            event.accept()


    @pyqtSlot()
    def onImportAudioClicked(self):
        try:
            audioDataPath, _ = QFileDialog.getOpenFileName(parent=self, 
                                                           caption='Import Audio Data', 
                                                           directory=self.window().rootDirectory,
                                                           options=QFileDialog.DontUseNativeDialog)
            if audioDataPath:
                self.audioDataPath.setText(audioDataPath)
        except Exception as e:
            self.window().emitToExceptionsManager(e)


    @pyqtSlot()
    def onTranscribeClicked(self):
        self.transcriptionResult.setText('Transcribing...')
        self.setDisabled(True)

        config = {
            'audioDataPath' : self.audioDataPath.text(),
            'encoding' : self.encoding.currentText(),
            'enhanced' : self.enhanced.checkState(),
            'languageCode' : self.language.currentText(),
            'model' : self.model.currentText(),
            'outputFolder' : self.window().getSetting('defaultOutputFolder'),
            'sampleRate' : self.sampleRate.value(),
            'serviceAccountPath' : self.window().getSetting('serviceAccountPath')
        }
        self.transcriptionController.resquestTranscription(config)


    @pyqtSlot(str)
    def onTranscriptionReady(self, transcription):
        self.transcriptionResult.setText(transcription)
        self.setDisabled(False)


    @pyqtSlot(Exception)
    def onException(self, e):
        self.window().emitToExceptionsManager(e)
        self.transcriptionResult.setText('')
        self.setDisabled(False)

from pathlib import Path

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QWidget

from src.app.application_container import ApplicationContainer
from src.app.controllers.transcription_controller import TranscriptionController
from src.app.gui.transcription_ui import Ui_Transcription



class Transcription(QWidget, Ui_Transcription):

    rootDirectory = str(Path(__file__).resolve().parents[3])

    def __init__(self, parent=None):
        super(Transcription, self).__init__(parent)
        self.setupUi(self)

        # Initilalization of the controller.
        self.transcriptionController = TranscriptionController()

        # Qt signal slots.
        self.btnImportAudio.clicked.connect(self.onImportAudioClicked)
        self.btnTranscribe.clicked.connect(self.onTranscribeClicked)
        self.transcriptionController.transcriptionReady.connect(self.onTranscriptionReady)
        self.transcriptionController.exception.connect(self.onException)


    # Handles the event where the user closes the window with the X button.
    def closeEvent(self, event):
        if event:
            self.transcriptionController.cancelTranscription()
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
            ApplicationContainer.exceptions().show(e)


    @pyqtSlot()
    def onTranscribeClicked(self):
        self.transcriptionResult.setText('Transcribing...')
        self.setDisabled(True)
        self.transcriptionController.requestTranscription(self.audioDataPath.text())


    @pyqtSlot(str)
    def onTranscriptionReady(self, transcription):
        self.transcriptionResult.setText(transcription)
        self.setDisabled(False)


    @pyqtSlot(Exception)
    def onException(self, e):
        ApplicationContainer.exceptions().show(e)
        self.transcriptionResult.setText('')
        self.setDisabled(False)


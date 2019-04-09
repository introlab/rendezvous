from pathlib import Path

from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot

from src.app.controllers.audio_processing_controller import AudioProcessingController
from src.app.gui.audio_processing_ui import Ui_AudioProcessing


class AudioProcessing(QWidget, Ui_AudioProcessing):

    def __init__(self, parent=None):
        super(AudioProcessing, self).__init__(parent)
        self.setupUi(self)

        # Initialization of the controller
        self.audioProcessingController = AudioProcessingController()

        # Populate UI
        self.cbNoiseReductionLib.addItems([noiseReductionLib.value for noiseReductionLib in self.audioProcessingController.getNoiseReductionLibs()])

        # Qt signal slots
        self.btnImportAudio.clicked.connect(self.onImportAudioClicked)
        self.btnProcessAudio.clicked.connect(self.onProcessAudioClicked)
        self.audioProcessingController.noiseReductionSignal.connect(self.onProcessAudioFinished)
        self.audioProcessingController.exception.connect(self.onException)


    # Handles the event where the user closes the window with the X button.
    def closeEvent(self, event):
        if event:
            event.accept()


    @pyqtSlot()
    def onProcessAudioClicked(self):
        noiseReductionLib = self.cbNoiseReductionLib.currentText()
        self.lblState.setText('Processing...')
        self.audioProcessingController.processNoiseReduction(noiseReductionLib, self.audioDataPath.text())
        self.btnProcessAudio.setDisabled(True)


    @pyqtSlot()
    def onProcessAudioFinished(self):
        self.btnProcessAudio.setDisabled(False)
        self.lblState.setText('Done!')


    @pyqtSlot()
    def onImportAudioClicked(self):
        try:
            audioDataPath, _ = QFileDialog.getOpenFileName(parent=self, 
                                                           caption='Import Audio Data', 
                                                           directory=self.window().rootDirectory,
                                                           options=QFileDialog.DontUseNativeDialog)
            if audioDataPath:
                if '.raw' in audioDataPath:
                    self.audioDataPath.setText(audioDataPath)
                else:
                    self.onException(Exception('Audio is not a .raw file'))
        except Exception as e:
            self.onException(e)


    @pyqtSlot(Exception)
    def onException(self, e):
        self.window().emitToExceptionsManager(e)
        self.btnProcessAudio.setDisabled(False)
        self.lblState.setText('Error')

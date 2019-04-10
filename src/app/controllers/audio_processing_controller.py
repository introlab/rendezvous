from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from src.app.services.noisereduction.noise_reduction import NoiseReduction, NoiseReductionLib


class AudioProcessingController(QObject):

    noiseReductionSignal = pyqtSignal()
    exception = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(AudioProcessingController, self).__init__(parent)

        # Initialization of the Noise Reduction service
        self.noiseReduction = NoiseReduction()
        self.noiseReductionThread = QThread()
        self.noiseReduction.moveToThread(self.noiseReductionThread)
        self.noiseReductionThread.started.connect(self.onProcessNoiseReduction)

        # Qt signal slots
        self.noiseReduction.noiseReductionSignal.connect(self.onNoiseReductionFinished)
        self.noiseReduction.exception.connect(self.onException)


    def processNoiseReduction(self, noiseReductionLib, audioPath):
        self.__noiseReductionLib = noiseReductionLib
        self.__audioPath = audioPath
        self.noiseReductionThread.start()


    def onProcessNoiseReduction(self):
        self.noiseReduction.reduceNoise(self.__noiseReductionLib, self.__audioPath)

    
    def getNoiseReductionLibs(self):
        return NoiseReductionLib


    @pyqtSlot()
    def onNoiseReductionFinished(self):
        self.noiseReductionSignal.emit()
        self.noiseReductionThread.quit()


    @pyqtSlot(Exception)
    def onException(self, e):
        self.noiseReductionThread.quit()
        self.exception.emit(e)

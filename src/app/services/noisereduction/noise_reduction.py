import copy
import numpy as np
from enum import Enum, unique

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from .noisereductionmethods.rnnoise import RNNoise


@unique
class NoiseReductionLib(Enum):
    RNNOISE = 'RNNoise'


class NoiseReduction(QObject):

    noiseReductionSignal = pyqtSignal()
    exception = pyqtSignal(Exception)

    def __init__(self):
        QObject.__init__(self)


    def __selectLib(self, noiseReductionLib):
        if noiseReductionLib == NoiseReductionLib.RNNOISE.value:
            self.__noiseReduction = RNNoise()
            self.__frameSize = self.__noiseReduction.getFrameSize()


    def reduceNoise(self, noiseReductionLib, audioPath):
        originalAudioData = None 
        try:
            self.__selectLib(noiseReductionLib)
            audioData = self.__readAudioFile(audioPath)
            # In case there's an exception, we rollback to the original file
            # originalAudioData = copy.deepcopy(audioData)
            
            denoisedData = np.empty([], 'int16')

            for i in range(0, len(audioData), self.__frameSize):
                frame = audioData[i:i + self.__frameSize]
                self.__noiseReduction.processFrame(frame)
                denoisedData = np.append(denoisedData, frame)

            self.__writeAudioFile(audioPath, denoisedData)
            self.noiseReductionSignal.emit()

        except Exception as e:
            # Uncomment when __writeAudioFile is modified
            # if originalAudioData is not None:
            #     self.__writeAudioFile(audioPath, originalAudioData)
            self.exception.emit(e)


    def __readAudioFile(self, audioPath):
        audioFile = open(audioPath, 'rb')
        audioData = np.fromfile(audioFile, np.int16)
        audioFile.close()

        return audioData


    def __writeAudioFile(self, audioPath, audioData):
        # temp: save result in a new file, eventually we want to replace the original by the denoised file
        audioPath = audioPath.replace('.raw', '')
        audioPath += 'Denoised.raw'

        audioFile = open(audioPath, 'w')
        audioData.tofile(audioFile)
        audioFile.close()

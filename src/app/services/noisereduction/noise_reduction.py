import numpy as np
from .noisereducer.rnnoise import RNNoise

class NoiseReduction():

    def __init__(self):
        self.__noiseReduction = RNNoise()
        self.__frameSize = self.__noiseReduction.getFrameSize()


    # audioData must be a int16 numpy array
    def reduceNoise(self, audioData):
        denoisedData = np.empty([], 'int16')
        #audioData = np.fromfile(soundFile, np.int16)

        for i in range(0, len(audioDataArray), self.__frameSize):
            frame = audioDataArray[i:i + self.__frameSize]
            self.__noiseReduction.processFrame(frame)
            denoisedData = np.append(denoisedData, frame)
        
        return denoisedData
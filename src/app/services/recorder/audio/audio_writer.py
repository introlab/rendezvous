import wave
import os
import numpy as np


class AudioWriter():

    def __init__(self, nChannels, byteDepth, sampleRate):
        self.nChannels = nChannels
        self.byteDepth = byteDepth
        self.sampleRate = sampleRate

    
    # Writes raw audio data into a file
    def writeRaw(self, data, path):

        if data.dtype == np.dtype('uint8'):
            audioFile = open(path, 'wb')
            data.tofile(audioFile)
            audioFile.close()
        else:
            raise Exception('The data used to generate a raw file is not a numpy uint8 array.')        

    
    # Takes a raw audio in mono and creates a wav file.
    def writeWav(self, data, path):

        if data.dtype == np.dtype('uint8'):
            wavFile = wave.open(path, 'wb')
            wavFile.setnchannels(self.nChannels)
            wavFile.setsampwidth(self.byteDepth)
            wavFile.setframerate(self.sampleRate)
            wavFile.writeframesraw(data)
            wavFile.close()
        else:
            raise Exception('The data used to generate a wav file is not a numpy uint8 array.')


    # Takes a raw audio in stereo and returns a raw audio in mono
    def rawStereoToMono(self, dataRaw):
        j = 0
        dataMono = np.empty([int(len(dataRaw/2))], np.int16)

        for i in range(0, len(dataRaw), 2):
            dataMono[j] = dataRaw[i]
            j += 1

        return dataMono


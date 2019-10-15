import wave
import os
import numpy as np
from src.app.services.recorder.audio.i_audio_output import IAudioOutput

class FileAudioOutput(IAudioOutput):
    '''
        Writer of .wav and .raw files. For .wav files, the number of channels, the number of byte per sample and the sample rate are needed.
    '''

    def __init__(self, nChannels, byteDepth, sampleRate):
        self.nChannels = nChannels
        self.byteDepth = byteDepth
        self.sampleRate = sampleRate
        self.__wavFile = None

    
    def openWav(self, path):
        self.__wavFile = wave.open(path, 'wb')
        self.__wavFile.setnchannels(self.nChannels)
        self.__wavFile.setsampwidth(self.byteDepth)
        self.__wavFile.setframerate(self.sampleRate)


    def closeWav(self):
        self.__wavFile.close()


    # Writes raw audio data into a file
    def writeRaw(self, data, path):

        if data.dtype == np.dtype('uint8'):
            audioFile = open(path, 'wb')
            data.tofile(audioFile)
            audioFile.close()
        else:
            raise Exception('The data used to generate a raw file is not a numpy uint8 array.')  

    
    # Takes a raw audio in mono and creates a wav file.
    def write(self, data):
        self.__wavFile.writeframesraw(data)


    # Takes a raw audio in stereo and returns a raw audio in mono
    def rawStereoToMono(self, dataRaw):
        j = 0
        dataMono = np.empty([int(len(dataRaw/2))], np.int16)

        for i in range(0, len(dataRaw), 2):
            dataMono[j] = dataRaw[i]
            j += 1

        return dataMono


import wave
import os

import soundfile


class AudioRecorder:

    def __init__(self):
        pass

    @staticmethod
    def convertRawToWav(rawPath):
        if not os.path.exists(rawPath):
            raise Exception('there is no file at : {path}'.format(path=rawPath))

        filename = os.path.splitext(rawPath)[0]
        data, samplerate = soundfile.read(rawPath, channels=1, samplerate=48000, format='RAW', subtype='FLOAT', endian='BIG')
        wavFile = '{filename}.wav'.format(filename=filename)
        soundfile.write(wavFile, data, samplerate, endian='BIG')
    
    
    def startRecording(self):
        pass

    
    def stopRecording(self):
        pass


if __name__ == '__main__':
    AudioRecorder.convertRawToWav('/home/morel/development/rendezvous/test_postfiltered.raw')


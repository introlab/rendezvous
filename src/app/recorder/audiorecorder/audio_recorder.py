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
        data, samplerate = soundfile.read(rawPath, channels=4, samplerate=48000, subtype='FLOAT')
        wavFile = '{filename}.wav'.format(filename=filename)
        soundfile.write(wavFile, data, samplerate)
    
    
    def startRecording(self):
        pass

    
    def stopRecording(self):
        pass


if __name__ == '__main__':
    AudioRecorder.convertRawToWav('test_postfiltered.raw')


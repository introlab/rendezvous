import wave
import os

from PyQt5.QtCore import QObject

class AudioWriter(QObject):

    def __init__(self, parent=None):
        self.isRecording = False

        self.wavFiles = []
        self.sourcesBuffer = {'0' :bytearray(), '1': bytearray(), '2': bytearray(), '3': bytearray()}


    def startRecording(self, outputFolder, nChannels, byteDepth, sampleRate):
        for i in range(0, 4):
            outputFile = os.path.join(outputFolder, 'outputsrc-{}.wav'.format(i))
            self.wavFiles.append(wave.open(outputFile, 'wb'))
            self.wavFiles[i].setnchannels(nChannels)
            self.wavFiles[i].setsampwidth(byteDepth)
            self.wavFiles[i].setframerate(sampleRate)

        self.isRecording = True


    def stopRecording(self):
        for index, wavFile in enumerate(self.wavFiles):
            if wavFile:
                audioRaw = self.sourcesBuffer[str(index)]
                wavFile.writeframesraw(audioRaw)
                wavFile.close()
                self.sourcesBuffer[str(index)] = bytearray()
        
        self.wavFiles = []
        self.isRecording = False

    
    def processRawData(self, rawBytes):
        if self.isRecording:
            nChannel = 4
            offset = 0
            while offset < len(rawBytes):
                for key, _ in self.sourcesBuffer.items():
                    currentByte = int(offset + int(key))
                    self.sourcesBuffer[key] += rawBytes[currentByte:currentByte + 2]

            offset += nChannel * 2


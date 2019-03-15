import wave
import os
from threading import Thread
import queue
from time import sleep

from PyQt5.QtCore import QObject, pyqtSignal

class AudioWriter(QObject, Thread):

    signalException = pyqtSignal(Exception)
    signalRecordingStarted = pyqtSignal()
    signalRecordingStopped = pyqtSignal()

    def __init__(self, parent=None):
        super(AudioWriter, self).__init__(parent)
        Thread.__init__(self)

        self.isRecording = False
        self.wavFiles = []
        self.outputFolder = ''
        self.nChannels = 4
        self.byteDepth = 2
        self.sampleRate = 48000
        self.sourcesBuffer = {'0' :bytearray(), '1': bytearray(), '2': bytearray(), '3': bytearray()}
        self.mailbox = queue.Queue()


    def changeWavSettings(self, outputFolder, nChannels, byteDepth, sampleRate):
        self.outputFolder = outputFolder
        self.nChannels = nChannels
        self.byteDepth = byteDepth
        self.sampleRate = sampleRate


    def stop(self):
        if self.isRecording:
            for index, wavFile in enumerate(self.wavFiles):
                if wavFile:
                    audioRaw = self.sourcesBuffer[str(index)]
                    wavFile.writeframesraw(audioRaw)
                    wavFile.close()
                    self.sourcesBuffer[str(index)] = bytearray()
            
            self.wavFiles = []
            
            self.mailbox.put('stop')
            # wait until the thread terminate
            self.join()
            self.isRecording = False
            self.signalRecordingStopped.emit()


    def run(self):
        try:
            for i in range(0, self.nChannels):
                outputFile = os.path.join(self.outputFolder, 'outputsrc-{}.wav'.format(i))
                self.wavFiles.append(wave.open(outputFile, 'wb'))
                self.wavFiles[i].setnchannels(self.nChannels)
                self.wavFiles[i].setsampwidth(self.byteDepth)
                self.wavFiles[i].setframerate(self.sampleRate)

            
            self.isRecording = True
            self.signalRecordingStarted.emit()
            while True:
                    data = self.mailbox.get()
                    if data == 'stop':
                        return
                    
                    nChannel = 4
                    offset = 0
                    while offset < len(data):
                        for key, _ in self.sourcesBuffer.items():
                            currentByte = int(offset + int(key))
                            self.sourcesBuffer[key] += data[currentByte:currentByte + 2]

                    offset += nChannel * 2
        
        except Exception as e:
            self.signalException.emit(e)


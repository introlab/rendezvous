import wave
import os
from threading import Thread
import queue
from enum import Enum, unique

from PyQt5.QtCore import QObject, pyqtSignal


@unique
class WriterActions(Enum):
    STOP = 'stop'
    SAVE_FILES = 'savefiles'
    NEW_RECORDING = 'newrecording'


class AudioWriter(QObject, Thread):

    signalException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(AudioWriter, self).__init__(parent)
        Thread.__init__(self)

        self.isRecording = False
        self.wavFiles = []
        self.outputFolder = ''
        self.nChannels = 4
        self.nChannelsFile = 1
        self.byteDepth = 2
        self.sampleRate = 48000
        self.sourcesBuffer = {'0' :bytearray(), '1': bytearray(), '2': bytearray(), '3': bytearray()}
        self.mailbox = queue.Queue()


    def changeWavSettings(self, outputFolder, nChannels, nChannelFile, byteDepth, sampleRate):
        self.outputFolder = outputFolder
        self.nChannels = nChannels
        self.nChannelsFile = nChannelFile
        self.byteDepth = byteDepth
        self.sampleRate = sampleRate


    def stop(self):
        if self.isRunning:
            self.mailbox.put(WriterActions.STOP)
            # Wait until the thread terminate.
            self.join()

            for wavFile in self.wavFiles:
                if wavFile:
                    wavFile.close()
        
            self.wavFiles = []
            self.isRunning = False


    def run(self):
        try:
            self.isRunning = True
            while True:
                    data = self.mailbox.get()

                    if isinstance(data, bytes):
                        offset = 0
                        while offset < len(data):
                            for key, _ in self.sourcesBuffer.items():
                                currentByte = int(offset + int(key))
                                self.sourcesBuffer[key] += data[currentByte:currentByte + self.byteDepth]

                            offset += self.nChannels * self.byteDepth

                    elif data == WriterActions.SAVE_FILES:
                        self.__writeWavFiles()

                    elif data == WriterActions.NEW_RECORDING:
                        for i in range(0, self.nChannels):
                            outputFile = os.path.join(self.outputFolder, 'outputsrc-{}.wav'.format(i))
                            self.wavFiles.append(wave.open(outputFile, 'wb'))
                            self.wavFiles[i].setnchannels(self.nChannelsFile)
                            self.wavFiles[i].setsampwidth(self.byteDepth)
                            self.wavFiles[i].setframerate(self.sampleRate)
                            self.sourcesBuffer[str(i)] = bytearray()

                    elif data == WriterActions.STOP:
                        break
        
        except Exception as e:
            self.signalException.emit(e)

        finally:
            self.isRunning = False


    def __writeWavFiles(self):
        for index, wavFile in enumerate(self.wavFiles):
            audioRaw = self.sourcesBuffer[str(index)]
            if wavFile and audioRaw:
                wavFile.writeframesraw(audioRaw)
                wavFile.close()
                self.sourcesBuffer[str(index)] = bytearray()
        
        self.wavFiles = []


import os
from threading import Thread
import queue
from enum import Enum, unique

from PyQt5.QtCore import QObject, pyqtSignal
from src.app.services.recorder.audio.audio_writer import AudioWriter
from src.app.services.recorder.video.video_writer import VideoWriter


@unique
class RecorderActions(Enum):
    STOP = 'stop'
    SAVE_FILES = 'savefiles'
    NEW_RECORDING = 'newrecording'

    
class Recorder(QObject, Thread):
    
    signalException = pyqtSignal(Exception)
    
    def __init__(self, parent=None):
        super(Recorder, self).__init__(parent)

        self.__audioWriter = AudioWriter()
        self.__videoWriter = VideoWriter()
        self.mailbox = queue.Queue()
        self.isRunning = False
        self.isRecording = False

    
    def stop(self):
        if self.isRunning:
            self.mailbox.put(RecorderActions.STOP)
            # Wait until the thread terminate.
            self.join()

            self.__audioWriter.close()
        
            self.wavFiles = []
            self.isRunning = False


    def run(self):
        try:
            self.isRunning = True
            while True:
                    data = self.mailbox.get()

                    if isinstance(data, bytes):
                        self.__audioWriter.write(data)

                    elif data == RecorderActions.SAVE_FILES:
                        self.__audioWriter.close()

                    elif data == RecorderActions.NEW_RECORDING:
                        self.__audioWriter.createNewFiles()

                    elif data == RecorderActions.STOP:
                        break
        
        except Exception as e:
            self.signalException.emit(e)

        finally:
            self.isRunning = False


    def changeAudioSettings(self, outputFolder, nChannels, nChannelFile, byteDepth, sampleRate):
        self.__audioWriter.changeWavSettings(outputFolder, nChannels, nChannelFile, byteDepth, sampleRate)


import wave
import os
from threading import Thread
from time import sleep

from PyQt5.QtCore import QObject, pyqtSignal

class AudioRecorder(QObject):

    signalException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(AudioRecorder, self).__init__(parent)
        self.isRunning = False
    
    
    def start(self, outputFolder):
        try:
            Thread(target=self.__run, args=[outputFolder]).start()
        
        except Exception as e:
            self.isRunning = False
            self.signalException.emit(e)
    

    def stop(self):
        self.isRunning = False


    def __run(self, outputFolder):
        try:
            pass

        except Exception as e:
            self.signalException(e)

        finally:
            self.stop()

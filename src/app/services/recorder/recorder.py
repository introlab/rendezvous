import os
import time
from threading import Thread
import queue
from enum import Enum, unique

from PyQt5.QtCore import QObject, pyqtSignal
from src.app.services.recorder.audio.audio_writer import AudioWriter
from src.app.services.recorder.video.video_writer import VideoWriter
from src.app.services.service.service_state import ServiceState

    
class Recorder(QObject, Thread):
    
    signalException = pyqtSignal(Exception)
    signalStateChanged = pyqtSignal(object)
    
    def __init__(self, outputFolder, nChannels, nChannelFile, byteDepth, sampleRate, parent=None):
        QObject.__init__(self, parent)
        Thread.__init__(self)

        self.mailbox = queue.Queue()

        self.__audioWriter = AudioWriter(outputFolder, nChannels, nChannelFile, byteDepth, sampleRate)
        self.__videoSource = VideoWriter(os.path.join(outputFolder, 'video-0.avi'), fourCC='MJPG', fps=10)

        self.__state = ServiceState.STOPPED


    def initialize(self):
        self.__state = ServiceState.STARTING
        self.signalStateChanged.emit(ServiceState.STARTING)
        
        super().start()

    
    def startRecording(self):
        self.__state = ServiceState.RUNNING
        self.signalStateChanged.emit(ServiceState.RUNNING)

    
    def stop(self):
        self.__state = ServiceState.STOPPING
        self.signalStateChanged.emit(ServiceState.STOPPING)

    
    def terminate(self):
        if self.__state != ServiceState.STOPPED and self.__state != ServiceState.TERMINATED:
            self.stop()
        else:
            self.__state = ServiceState.TERMINATED


    def run(self):

        try:

            self.__state = ServiceState.READY
            self.signalStateChanged.emit(ServiceState.READY)

            print('Recorder started')

            while self.__state == ServiceState.RUNNING or self.__state == ServiceState.READY:
                
                if self.__state == ServiceState.READY:
                    time.sleep(0.5)
                    self.signalStateChanged.emit(ServiceState.READY)
                    continue

                data = None
                try:
                    data = self.mailbox.get_nowait()
                except queue.Empty:
                    time.sleep(0.0001)

                if data is not None:

                    dataType = data[0]
                    if dataType == 'audio':
                        self.__audioWriter.write(data[1])

                    elif dataType == 'video':
                        self.__videoSource.write(data[1])

                    else:
                        raise Exception('data type {type} not supported for recording'.format(dataType))

        except Exception as e:
            
            self.__state = ServiceState.STOPPING   
            self.signalStateChanged.emit(ServiceState.STOPPING)          
            self.signalException.emit(e)

        finally:

            # Empty the queue
            try:
                while True:
                    queue.get_nowait()
            except:
                pass

            self.__audioWriter.close()
            self.__videoSource.close()

            self.signalStateChanged.emit(ServiceState.STOPPED)
            self.__state = ServiceState.STOPPED

            while self.__state == ServiceState.STOPPED:
                time.sleep(0.5)
                self.signalStateChanged.emit(ServiceState.STOPPED)

            self.signalStateChanged.emit(ServiceState.TERMINATED)

            print('Recorder terminated')

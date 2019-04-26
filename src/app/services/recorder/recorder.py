import os
import time
import queue
from threading import Thread
from enum import Enum, unique

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal
from src.app.services.recorder.audio.audio_writer import AudioWriter
from src.app.services.recorder.video.video_writer import VideoWriter
from src.app.services.service.service_state import ServiceState
from src.utils.media_merger import MediaMerger

    
class Recorder(QObject, Thread):
    '''
        Recording thread based on a queue for recording both sounds and videos.
        Based on SourceClassifier class, it decides if the audio/video source is human or not.
        If it is, the source is kept and the data associated to this source is recorded.
    '''
    
    signalException = pyqtSignal(Exception)
    signalStateChanged = pyqtSignal(object)
    
    def __init__(self, outputFolder, nChannels, byteDepth, sampleRate, parent=None):
        QObject.__init__(self, parent)
        Thread.__init__(self)

        self.mailbox = queue.Queue()

        self.__audioStereoPath = os.path.join(outputFolder, 'audioStereo.wav')
        self.__audioMonoPath = os.path.join(outputFolder, 'media.wav')
        self.__videoPath = os.path.join(outputFolder, 'video.avi')
        self.__mediaPath = os.path.join(outputFolder, 'media.avi')
        
        self.__audioWriter = AudioWriter(nChannels, byteDepth, sampleRate)
        self.__videoSource = VideoWriter(self.__videoPath, fourCC='MJPG', fps=15)

        self.state = ServiceState.TERMINATED

        self.humanSourcesIndex = []

        self.__bufferSize = 102400
        self.__humanSourcesBuff = np.empty(self.__bufferSize, np.uint8)
        self.__currentBufferIndex = 0


    def initialize(self):
        self.state = ServiceState.STARTING
        self.signalStateChanged.emit(ServiceState.STARTING)
        
        super().start()


    def startRecording(self):
        self.state = ServiceState.RUNNING
        self.signalStateChanged.emit(ServiceState.RUNNING)


    def stop(self):
        self.state = ServiceState.STOPPING
        self.signalStateChanged.emit(ServiceState.STOPPING)


    def terminate(self):
        if self.state != ServiceState.STOPPED and self.state != ServiceState.TERMINATED:
            self.stop()
        else:
            self.state = ServiceState.TERMINATED


    # Keeps only human sources
    def separateSourcesData(self, data):
        # We receive chunks of 1024 bytes
        byteMask = self.createMaskFromHumanSources(1024*8)
        byteArray = np.frombuffer(data, np.uint8)

        filteredHumanSources = np.bitwise_and(byteArray, byteMask)
        
        self.__humanSourcesBuff[self.__currentBufferIndex : self.__currentBufferIndex + len(filteredHumanSources)] = filteredHumanSources
        self.__currentBufferIndex += len(filteredHumanSources)


    def createMaskFromHumanSources(self, length):   
            byteLength = int(length/8)
            mask = np.zeros(byteLength, np.uint8)

            self.humanSourcesIndex = [0, 1, 2, 3]
            
            if 0 in self.humanSourcesIndex:
                for i in range(0, byteLength-1, 8):
                    mask[i] = 255
                    mask[i+1] = 255
            if 1 in self.humanSourcesIndex:
                for i in range(2, byteLength-1, 8):
                    mask[i] = 255
                    mask[i+1] = 255
            if 2 in self.humanSourcesIndex:
                for i in range(4, byteLength-1, 8):
                    mask[i] = 255
                    mask[i+1] = 255
            if 3 in self.humanSourcesIndex:
                for i in range(6, byteLength-1, 8):
                    mask[i] = 255
                    mask[i+1] = 255

            return mask


    def recordData(self, data):
        dataType = data[0]
        if dataType == 'audio':
            self.separateSourcesData(data[1])

            if self.__currentBufferIndex == self.__bufferSize:
                self.__audioWriter.writeWavFrame(self.__humanSourcesBuff)
                self.__currentBufferIndex = 0

        elif dataType == 'video':
            self.__videoSource.write(data[1])

        else:
            raise Exception('data type {type} not supported for recording'.format(dataType))
   

    def run(self):

        try:
            
            self.__audioWriter.openWav(self.__audioStereoPath)
            
            self.state = ServiceState.READY
            self.signalStateChanged.emit(ServiceState.READY)

            while self.state == ServiceState.RUNNING or self.state == ServiceState.READY:

                if self.state == ServiceState.READY:
                    time.sleep(0.5)
                    self.signalStateChanged.emit(ServiceState.READY)
                    continue

                data = None
                try:
                    data = self.mailbox.get_nowait()
                except queue.Empty:
                    time.sleep(0.0001)

                if data is not None:
                    self.recordData(data)

        except Exception as e:

            self.signalException.emit(e)
            self.state = ServiceState.STOPPING   
            self.signalStateChanged.emit(ServiceState.STOPPING)          

        finally:
            
            # Empty the queue
            try:
                while True:
                    data = None
                    data = self.mailbox.get_nowait()

                    if data is not None:
                        self.recordData(data)

            except:
                pass

            self.__audioWriter.writeWavFrame(self.__humanSourcesBuff[:self.__currentBufferIndex])

            self.__videoSource.close()
            self.__audioWriter.closeWav()

            if os.path.exists(self.__audioStereoPath) and os.path.exists(self.__videoPath):
                print('Merging audio and video...')

                # Convert stereo into mono
                MediaMerger.stereoToMono(self.__audioStereoPath, self.__audioMonoPath)

                # Merge audio-video-text
                MediaMerger.mergeFiles(self.__audioMonoPath, self.__videoPath, self.__mediaPath)

                print('Merge done!')

            self.signalStateChanged.emit(ServiceState.STOPPED)
            self.state = ServiceState.STOPPED

            while self.state == ServiceState.STOPPED:
                time.sleep(0.5)
                self.signalStateChanged.emit(ServiceState.STOPPED)

            self.signalStateChanged.emit(ServiceState.TERMINATED)

            print('Recorder terminated')


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
    
    def __init__(self, outputFolder, parent=None):
        super(Recorder, self).__init__(parent)

        self.__audioWriter = AudioWriter()
        self.__videoSources = {
            '0': VideoWriter(os.path.join(outputFolder, 'video-0.avi'), fourCC='MJPG', fps=20),
            '1': VideoWriter(os.path.join(outputFolder, 'video-1.avi'), fourCC='MJPG', fps=20),
            '2': VideoWriter(os.path.join(outputFolder, 'video-2.avi'), fourCC='MJPG', fps=20),
            '3': VideoWriter(os.path.join(outputFolder, 'video-3.avi'), fourCC='MJPG', fps=20)
        }
        self.mailbox = queue.Queue()
        self.isRunning = False
        self.isRecording = False

    
    def stop(self):
        if self.isRunning:
            self.mailbox.put(RecorderActions.STOP)
            # Wait until the thread terminate.
            self.join()

            self.__audioWriter.close()
            for key, videoWriter in self.__videoSources.items():
                videoWriter.close()
        
            self.isRunning = False


    def run(self):
        try:
            self.isRunning = True
            while True:
                    data = self.mailbox.get()

                    if isinstance(data, tuple):
                        dataType = data[0]
                        if dataType == 'audio':
                            self.__audioWriter.write(data[1])

                        elif dataType == 'video':
                            sourceIndex = data[1]

                            if sourceIndex in self.__videoSources.keys():
                                self.__videoSources[sourceIndex].write(data[2])

                            else:
                                fileFullPath = os.path.join(outputFolder, 'video-{}.avi'.format(str(sourceIndex)))
                                self.__videoSources[sourceIndex] = VideoWriter(fileFullPath, fourCC='MJPG', fps=20)
                                self.__videoSources[sourceIndex].write(data[2])

                        else:
                            raise Exception('data type not supported for recording')

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

    
    def changeVideoSettings(self, outputFolder):
        for key, writer in self.__videoSources.items():
            writer.setFilePath(os.path.join(outputFolder, 'video-{}.avi'.format(str(key))))


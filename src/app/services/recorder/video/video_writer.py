from threading import Thread
import numpy as np
import cv2

from PyQt5.QtCore import QObject, pyqtSignal

class VideoWriter(QObject):

    def __init__(self, filepath, fourCC, fps, parent=None):
        super(VideoWriter, self).__init__(parent)
        
        self.filepath = filepath
        self.fourCC = cv2.VideoWriter_fourcc(*fourCC)
        self.fps = fps
        self.__writer = None


    def write(self, frame):
        if not self.__writer:
            height, width, _ = frame.shape
            self.__writer = cv2.VideoWriter(self.filepath, self.fourCC, self.fps, (width, height))
        
        self.__writer.write(frame)
    

    def close(self):
        if self.__writer:
            self.__writer.release()

import os
from threading import Thread
import numpy as np
import cv2

from PyQt5.QtCore import QObject, pyqtSignal

class VideoWriter(QObject):

    def __init__(self, parent=None):
        super(VideoWriter, self).__init__(parent)


if __name__ == '__main__':

    frameWidth = self.virtualCameraFrame.size().width()
    frameHeight = self.virtualCameraFrame.size().height()
    writer = cv2.VideoWriter('video.avi', 'MJPEG', '20', (frameWidth, frameHeight))
    writer.release()


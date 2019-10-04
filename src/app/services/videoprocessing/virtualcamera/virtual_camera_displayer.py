import math

from PyQt5.QtGui import QImage, QPainter
from PyQt5 import QtCore


class VirtualCameraDisplayer:

    def __init__(self, frame):
        self.virtualCameraFrame = frame
        self.virtualCameraFrame.paintEvent = self.__paintEvent
        self.image = []
        self.isDisplaying = False


    def startDisplaying(self):
        self.isDisplaying = True


    def stopDisplaying(self):
        self.isDisplaying = False
        self.image = []
        self.virtualCameraFrame.update()


    def updateDisplay(self, image):
        if self.isDisplaying:
            print("update display")
            self.image = image
            self.virtualCameraFrame.update()


    # Draw every virtual cameras on the frame.
    # Is called automatically by Qt when the frame is ready for an update
    def __paintEvent(self, event):
        if not self.isDisplaying or self.image == []:
            return

        # Draw the image on screen
        painter = QPainter(self.virtualCameraFrame)
        width = self.virtualCameraFrame.size().width()
        height = self.virtualCameraFrame.size().height()
        resized = self.__createQImageFromOpenCVImage(self.image).scaled(width, height, QtCore.Qt.KeepAspectRatio)
        painter.drawImage(self.virtualCameraFrame.size().width() / 2 - resized.width() / 2, 0, resized)


    def __createQImageFromOpenCVImage(self, image):
        imageHeight, imageWidth, colors = image.shape
        bytesPerLine = 3 * imageWidth
        qImage = QImage(image.data, imageWidth, imageHeight, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        return qImage

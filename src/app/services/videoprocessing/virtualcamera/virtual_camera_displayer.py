import math

from PyQt5.QtGui import QImage, QPainter


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
            self.image = image
            self.virtualCameraFrame.update()


    # Draw every virtual cameras on the frame.
    # Is called automatically by Qt when the frame is ready for an update
    def __paintEvent(self, event):
        if not self.isDisplaying or self.image == []:
            return

        # Draw the image on screen
        painter = QPainter(self.virtualCameraFrame)
        painter.drawImage(0, 0, self.__createQImageFromOpenCVImage(self.image))


    def __createQImageFromOpenCVImage(self, image):
        imageHeight, imageWidth, colors = image.shape
        bytesPerLine = 3 * imageWidth
        qImage = QImage(image.data, imageWidth, imageHeight, bytesPerLine, QImage.Format.RGB888).rgbSwapped()
        return qImage

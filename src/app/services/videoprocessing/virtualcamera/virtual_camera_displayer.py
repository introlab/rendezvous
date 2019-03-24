import math

from PyQt5.QtGui import QImage, QPainter


class VirtualCameraDisplayer:

    def __init__(self, frame):
        self.virtualCameraFrame = frame
        self.virtualCameraFrame.paintEvent = self.__paintEvent

        self.spacing = 10

        self.image = []
        self.virtualCameras = []

    
    def updateDisplay(self, openCVImage, virtualCameras):
        self.image = openCVImage
        self.virtualCameras = virtualCameras

    
    def clear(self):
        self.image = []
        self.virtualCameras = []


    def clear(self):
        self.image = []
        self.virtualCameras = []


    # Draw every virtual cameras on the frame.
    # Is called automatically by Qt when the frame is ready for an update
    def __paintEvent(self, event):
        numberOfVCs = len(self.virtualCameras)
        if numberOfVCs == 0 or self.image == []:
            self.virtualCameraFrame.update()
            return
        
        displayPositions, vcWidth, vcHeight = self.__buildDisplay(numberOfVCs)
        painters = []
        for i in range(0, numberOfVCs):
            (xPos, yPos) = displayPositions[i] 

            vc = self.virtualCameras[i]
            (x1, y1, x2, y2) = vc.getBoundingRect()
            
            # Crops the image to the virtual camera
            croppedImage = self.image[y1:y2, x1:x2].copy()
            qImage = self.__createQImageFromOpenCVImage(croppedImage)

            # Draw the image on screen
            painter = QPainter(self.virtualCameraFrame)
            painter.drawImage(xPos - math.floor(vcWidth / 2), yPos - math.floor(vcHeight / 2), qImage.scaled(vcWidth, vcHeight))
            painters.append(painter)

        # This must be called to apply changes
        self.virtualCameraFrame.update()


    def __createQImageFromOpenCVImage(self, image):
        imageHeight, imageWidth, colors = image.shape
        bytesPerLine = 3 * imageWidth
        qImage = QImage(image.data, imageWidth, imageHeight, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        return qImage


    # Finds the position of every virtual camera along with their width and height.
    # This will surely change when weights display will be implemented.
    def __buildDisplay(self, numberOfVCs):
        frameWidth = self.virtualCameraFrame.size().width()
        frameHeight = self.virtualCameraFrame.size().height()

        availableWidth = frameWidth - (self.spacing * (numberOfVCs - 1))
        availableHeight = frameHeight - 50

        maxWidth = math.floor(availableWidth / numberOfVCs)
        newHeight = maxWidth * 4 / 3
        
        if newHeight > availableHeight:
            heightExcess = newHeight - availableHeight
            newHeight -= heightExcess
            maxWidth = newHeight * 3 / 4

        vcWidth = maxWidth
        vcHeight = vcWidth * 4 / 3

        freeWidth = frameWidth - numberOfVCs * vcWidth - (numberOfVCs - 1) * self.spacing

        positions = []
        for i in range(numberOfVCs):
            xPos = math.floor(freeWidth / 2 + i * (self.spacing + vcWidth) + vcWidth / 2)
            yPos = math.floor(frameHeight / 2)
            positions.append((xPos, yPos))

        return positions, vcWidth, vcHeight
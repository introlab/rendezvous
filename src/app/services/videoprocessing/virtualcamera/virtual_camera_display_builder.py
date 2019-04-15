import numpy as np
import math
import cv2


class VirtualCameraDisplayBuilder:

    def __init__(self):
        pass


    @staticmethod
    def buildImage(images, frameSize, backgroundColor, spacing):
        frameWidth = frameSize.width() - 2
        frameHeight = frameSize.height() - 2

        displayPositions, vcWidth, vcHeight = VirtualCameraDisplayBuilder.buildDisplay(len(images), frameWidth, frameHeight, spacing)
        
        combinedImage = np.full((frameHeight, frameWidth, 3), backgroundColor.red(), np.uint8)

        for i in range(0, len(images)):
            resized = cv2.resize(images[i], (vcWidth, vcHeight))
            (xPos, yPos) = displayPositions[i]
            xPos = xPos - math.floor(vcWidth / 2)
            yPos = yPos - math.floor(vcHeight / 2)
            combinedImage[yPos:yPos+vcHeight, xPos:xPos+vcWidth] = resized

        return combinedImage


    # Finds the position of every virtual camera along with their width and height.
    # This will surely change when weights display will be implemented.
    @staticmethod
    def buildDisplay(numberOfVCs, frameWidth, frameHeight, spacing):
        if numberOfVCs < 1:
            return [], 0, 0

        availableWidth = frameWidth - (spacing * (numberOfVCs - 1))
        availableHeight = frameHeight - 50

        maxWidth = math.floor(availableWidth / numberOfVCs)
        newHeight = maxWidth * 4 / 3
        
        if newHeight > availableHeight:
            heightExcess = newHeight - availableHeight
            newHeight -= heightExcess
            maxWidth = newHeight * 3 / 4

        vcWidth = math.floor(maxWidth)
        vcHeight = math.floor(vcWidth * 4 / 3)

        freeWidth = frameWidth - numberOfVCs * vcWidth - (numberOfVCs - 1) * spacing

        positions = []
        for i in range(numberOfVCs):
            xPos = math.floor(freeWidth / 2 + i * (spacing + vcWidth) + vcWidth / 2)
            yPos = math.floor(frameHeight / 2)
            positions.append((xPos, yPos))

        return positions, vcWidth, vcHeight
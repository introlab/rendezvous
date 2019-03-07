from PyQt5.QtCore import QTimer

from .virtual_camera import VirtualCamera
from src.utils.geometry_utils import GeometryUtils


class VirtualCameraManager:

    def __init__(self):
        self.virtualCameras = []

        # 3:4 (portrait)
        self.aspectRatio = 3 / 4

        # Distance from existing virtual cameras at which new virtual cameras are created,
        # If it is closer, it means we detected an existing virtual camera
        self.newCameraPositionThreshold = 150

        # Change in position that cause a move of the virtual camera
        self.positionChangedThreshold = 10

        # Change in dimension that cause a resize of the virtual camera
        self.dimensionChangeThreshold = 15

        # Factors to smooth out movements and resizing of virtual cameras
        # Smaller factor means smoother but slower movements
        self.resizeSmoothingFactor = 1 / 4
        self.moveSmoothingFactor = 1 / 2

        # Face scale factor to get the person's portrait (with shoulders)
        self.portraitScaleFactor = 2.5

        # Garbage collector unused virtual cameras. Ticks every second
        self.timer = QTimer()
        self.timer.timeout.connect(self.__garbageCollect)
        self.timer.start(1000)


    # Takes a face detection and tries to find an existing virtual camera to match it with.
    # If none is found, a new virtual camera is created.
    def addOrUpdateVirtualCamera(self, face, imageWidth, imageHeight):
        # We ignore the faces that overflow the image
        (xOverflow, yOverflow) = self.__findOverflow(face, imageWidth, imageHeight)     
        if xOverflow != 0  or yOverflow != 0:
            return

        newVirtualCamera = VirtualCamera.createFromFace(face)
        self.__tryResizeVirtualCamera(newVirtualCamera, self.portraitScaleFactor, imageWidth, imageHeight)

        existingVirtualCamera = self.__findExistingVirtualCamera(newVirtualCamera)
        if existingVirtualCamera:
            existingVirtualCamera.resetTimeToLive()
            self.__updateVirtualCamera(existingVirtualCamera, newVirtualCamera, imageWidth, imageHeight)
        else:
            self.virtualCameras.append(newVirtualCamera)


    # Compares the new virtual camera to existing ones to find a match
    def __findExistingVirtualCamera(self, newVirtualCamera):
        for vc in self.virtualCameras:
            distance = GeometryUtils.distanceBetweenTwoPoints(newVirtualCamera.getPosition(), vc.getPosition())
            if distance < self.newCameraPositionThreshold:
                return vc
        return None


    # Updates the size and position of the virtual camera if it has changed
    def __updateVirtualCamera(self, existingVirtualCamera, newVirtualCamera, imageWidth, imageHeight):
        if abs(newVirtualCamera.width - existingVirtualCamera.width) > self.dimensionChangeThreshold and \
          abs(newVirtualCamera.height - existingVirtualCamera.height) > self.dimensionChangeThreshold:
            dw = newVirtualCamera.width - existingVirtualCamera.width
            resizeFactor = (existingVirtualCamera.width + dw * self.resizeSmoothingFactor) / existingVirtualCamera.width
            self.__tryResizeVirtualCamera(existingVirtualCamera,
                                          resizeFactor,
                                          imageWidth,
                                          imageHeight)

        distance = GeometryUtils.distanceBetweenTwoPoints(existingVirtualCamera.getPosition(), newVirtualCamera.getPosition())
        if distance > self.positionChangedThreshold:
            x, y = existingVirtualCamera.getPosition()
            newX, newY = newVirtualCamera.getPosition()
            (unitX, unitY) = GeometryUtils.getUnitVector(newX - x, newY - y)
            smoothDx = distance * self.moveSmoothingFactor * unitX
            smoothDy = distance * self.moveSmoothingFactor * unitY
            self.__tryMoveVirtualCamera(existingVirtualCamera,
                                        (existingVirtualCamera.xPos + smoothDx, existingVirtualCamera.yPos + smoothDy),
                                        imageWidth,
                                        imageHeight)

     
    # Try to move the vc to the desired position. If a move in a certain dimension
    # makes the vc overflow the image, we disallow that move
    def __tryMoveVirtualCamera(self, virtualCamera, newPosition, imageWidth, imageHeight):
        (newPosX, newPosY) = newPosition
        virtualCamera.xPos = newPosX
        virtualCamera.yPos = newPosY

        (xOverflow, yOverflow) = self.__findOverflow(virtualCamera.getBoundingRect(), imageWidth, imageHeight)

        # There is an overflow, we need to remove it so the vc does not exit the image
        if xOverflow != 0:
            virtualCamera.xPos -= xOverflow
        if yOverflow != 0:
            virtualCamera.yPos -= yOverflow


    # Try to resize the vc with the desired scale factor. 
    # If the new dimensions overflow the image, we move the image to remove the overflow
    def __tryResizeVirtualCamera(self, virtualCamera, scaleFactor, imageWidth, imageHeight):
        # Find the desired dimensions respecting the aspect ratio
        virtualCamera.height = min(virtualCamera.height * scaleFactor, imageHeight * 0.9)
        virtualCamera.width = virtualCamera.height * self.aspectRatio

        # Remove caused overflow, if any
        self.__tryMoveVirtualCamera(virtualCamera, virtualCamera.getPosition(), imageWidth, imageHeight)


    # Finds by how much the rectangle overflows the image
    def __findOverflow(self, rect, imageWidth, imageHeight):
        (x1, y1, x2, y2) = rect
        xOverflow = 0
        yOverflow = 0

        if x1 < 0:
            xOverflow = x1
        elif imageWidth - x2 < 0:
            xOverflow = x2 - imageWidth

        if y1 < 0:
            yOverflow = y1
        elif imageHeight - y2 < 0:
            yOverflow = y2 - imageHeight

        return (xOverflow, yOverflow)


    # Removes the virtual cameras that were not associated with a face for some time
    def __garbageCollect(self):
        for vc in self.virtualCameras:
            vc.decreaseTimeToLive()
            if not vc.isAlive():
                self.virtualCameras.remove(vc)

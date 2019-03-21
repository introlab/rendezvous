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
        self.newCameraPositionThreshold = 50

        # Change in position that cause a move of the virtual camera
        self.positionChangedThreshold = 10

        # Change in dimension that cause a resize of the virtual camera
        self.dimensionChangeThreshold = 10

        # Factors to smooth out movements and resizing of virtual cameras
        # Smaller factor means smoother but slower movements
        self.resizeSmoothingFactor = 1 / 4
        self.moveSmoothingFactor = 1 / 2

        # Face scale factor to get the person's portrait (with shoulders)
        self.portraitScaleFactor = 3

        # Garbage collector unused virtual cameras. Ticks every second
        self.timer = QTimer()
        self.timer.timeout.connect(self.__garbageCollect)
        self.timer.start(1000)


    def update(self, faces, imageWidth, imageHeight):

        # Find matches between existing virtual cameras and detected faces
        matches = dict.fromkeys(self.virtualCameras, [])
        matches, unmatchedFaces = self.__tryFindMatches(matches, faces)

        # Create new virtual cameras from faces that were not associated with a vc
        for unmatchedFace in unmatchedFaces:
            newVirtualCamera = VirtualCamera.createFromFace(unmatchedFace)
            self.__tryResizeVirtualCamera(newVirtualCamera, self.portraitScaleFactor, imageWidth, imageHeight)
            self.virtualCameras.append(newVirtualCamera)

        for vc, face in matches.items():       
            # Found a face to associate the vc with, so we update the vc with its matched face
            if face != []:
                vc.resetTimeToLive()
                self.__updateVirtualCamera(vc, face, imageWidth, imageHeight)


    # Updates the size and position of the virtual camera if it has changed
    def __updateVirtualCamera(self, existingVirtualCamera, face, imageWidth, imageHeight):

        # Create virtual camera from face to compare with the vc we are updating
        newVirtualCamera = VirtualCamera.createFromFace(face)
        self.__tryResizeVirtualCamera(newVirtualCamera, self.portraitScaleFactor, imageWidth, imageHeight)

        if abs(newVirtualCamera.width - existingVirtualCamera.width) > self.dimensionChangeThreshold and \
           abs(newVirtualCamera.height - existingVirtualCamera.height) > self.dimensionChangeThreshold:
            dw = newVirtualCamera.width - existingVirtualCamera.width
            resizeFactor = (existingVirtualCamera.width + dw * self.resizeSmoothingFactor) / existingVirtualCamera.width
            self.__tryResizeVirtualCamera(existingVirtualCamera,
                                          resizeFactor, imageWidth,
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


    # Recursive method to find matches between virtual cameras and detected faces.
    # Returns vc-face matches plus all of the unmatched faces.
    def __tryFindMatches(self, matches, unmatchedFaces):

        # Returns if there is no unmatched vc's left or no unmatched faces left 
        noMatches = {vc: face for vc, face in matches.items() if face == []}
        if len(noMatches) < 1 or len(unmatchedFaces) < 1:
            return matches, unmatchedFaces

        # Map containing the closest vc to a face by distance
        faceDistanceVcMap = {}
        for vc, face in noMatches.items():  
            closestFaceIndex, distance = self.__findClosestFace(vc, unmatchedFaces)
            if closestFaceIndex is not None:
                currentFaceBestDistance, currentBestVc = faceDistanceVcMap.get(closestFaceIndex, (None, None))
                if currentFaceBestDistance is None or distance < currentFaceBestDistance:
                    faceDistanceVcMap[closestFaceIndex] = (distance, vc)
            
        # Update the matches with the closest face to the vc
        facesToMatch = unmatchedFaces
        for closestFaceIndex in faceDistanceVcMap.keys(): 
            (distance, vc) = faceDistanceVcMap[closestFaceIndex]
            closestFace = facesToMatch[closestFaceIndex]
            matches[vc] = closestFace
            unmatchedFaces = [face for face in unmatchedFaces if not (face==closestFace).all()]

        # Call the method recursively to match the remaining unmatched vc's
        return self.__tryFindMatches(matches, unmatchedFaces)


    # Find the closest face to the specified virtual camera
    def __findClosestFace(self, virtualCamera, faces):
        closestIndex = None
        closestDistance = None

        for i in range(len(faces)):
            (facex1, facey1, facex2, facey2) = faces[i]
            facePosition = (facex1 + (facex2 - facex1) / 2, facey1 + (facey2 - facey1) / 2)

            distance = GeometryUtils.distanceBetweenTwoPoints(virtualCamera.getPosition(), facePosition)
            if not closestDistance or distance < closestDistance:
                closestDistance = distance
                closestIndex = i

        if closestIndex is not None:
            return closestIndex, closestDistance
        else:
            return None, None


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

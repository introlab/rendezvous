from PyQt5.QtCore import QTimer

from .virtual_camera import VirtualCamera
from src.utils.geometry_utils import GeometryUtils


class VirtualCameraManager:

    def __init__(self):
        self.__virtualCameras = []

        self.virtualCameraMinHeight = 300

        # 3:4 (portrait)
        self.aspectRatio = 3 / 4

        # Change in position that cause a move of the virtual camera
        self.positionChangedThreshold = 25

        # Change in dimension that cause a resize of the virtual camera
        self.dimensionChangeThreshold = 30

        # Face scale factor to get the person's portrait (with shoulders)
        self.portraitScaleFactor = 5

        # Garbage collector unused virtual cameras. Ticks every second
        self.timer = QTimer()
        self.timer.timeout.connect(self.__garbageCollect)
        self.timer.start(1000)


    # Updates virtual camera move and resize animations every frame based on time since last frame
    def update(self, frameTime, imageWidth, imageHeight):      
        for vc in self.__virtualCameras:
            distance = GeometryUtils.distanceBetweenTwoPoints(vc.getPosition(), vc.positionGoal)
            if distance > self.positionChangedThreshold:    
                x, y = vc.getPosition()
                goal = vc.positionGoal
                distx = goal[0] - x
                disty = goal[1] - y
                dx = distx * frameTime * 2
                dy = disty * frameTime * 2
                self.__tryMoveVirtualCamera(vc,
                                            (vc.xPos + dx, vc.yPos + dy),
                                            imageWidth,
                                            imageHeight)

            if abs(vc.sizeGoal[0] - vc.width)  > self.dimensionChangeThreshold and \
               abs(vc.sizeGoal[1] - vc.height) > self.dimensionChangeThreshold:
                dw = vc.sizeGoal[1] - vc.height
                resizeFactor = (vc.height + dw * frameTime) / vc.height
                self.__tryResizeVirtualCamera(vc,
                                              resizeFactor, 
                                              imageWidth,
                                              imageHeight)


    # Updates the associated face (if any) for each virtual camera
    def updateFaces(self, faces, imageWidth, imageHeight):
        # Find matches between existing virtual cameras and detected faces
        matches = dict.fromkeys(self.__virtualCameras, None)
        matches, unmatchedFaces = self.__tryFindMatches(matches, faces)

        # Create new virtual cameras from faces that were not associated with a vc
        for unmatchedFace in unmatchedFaces:
            newVirtualCamera = VirtualCamera.createFromFace(unmatchedFace)
            ratio = max(self.portraitScaleFactor, self.virtualCameraMinHeight / newVirtualCamera.height)
            self.__tryResizeVirtualCamera(newVirtualCamera, ratio, imageWidth, imageHeight)
            newVirtualCamera.sizeGoal = (newVirtualCamera.width, newVirtualCamera.height)
            self.__virtualCameras.append(newVirtualCamera)

        for vc, face in matches.items():       
            # Found a face to associate the vc with, so we update the vc with its matched face
            if face:
                vc.resetTimeToLive()
                vc.positionGoal = face.getPosition()
                heightGoal = max(self.virtualCameraMinHeight, face.height * self.portraitScaleFactor)
                vc.sizeGoal = (heightGoal * self.aspectRatio, heightGoal)


    # Returns a copy of the virtual cameras so the caller can't modify the original ones
    def getVirtualCameras(self):
        vcs = []
        for vc in self.__virtualCameras:
            vcs.append(VirtualCamera.copy(vc))
        return vcs


    def clear(self):
        self.__virtualCameras.clear()

     
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
        virtualCamera.height = min(virtualCamera.height * scaleFactor, imageHeight)
        virtualCamera.width = virtualCamera.height * self.aspectRatio

        # Remove caused overflow, if any
        self.__tryMoveVirtualCamera(virtualCamera, virtualCamera.getPosition(), imageWidth, imageHeight)


    # Recursive method to find matches between virtual cameras and detected faces.
    # Returns vc-face matches plus all of the unmatched faces.
    def __tryFindMatches(self, matches, unmatchedFaces):

        # Returns if there is no unmatched vc's left or no unmatched faces left 
        noMatches = {vc: face for vc, face in matches.items() if face is None}
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
            unmatchedFaces = [face for face in unmatchedFaces if not face == closestFace]

        # Call the method recursively to match the remaining unmatched vc's
        return self.__tryFindMatches(matches, unmatchedFaces)


    # Find the closest face to the specified virtual camera
    def __findClosestFace(self, virtualCamera, faces):
        closestIndex = None
        closestDistance = None

        for i in range(len(faces)):
            facePosition = faces[i].getPosition()
            distance = GeometryUtils.distanceBetweenTwoPoints(virtualCamera.getPosition(), facePosition)
            if not closestDistance or distance < closestDistance:
                closestDistance = distance
                closestIndex = i

        if closestIndex is not None:
            return closestIndex, closestDistance
        else:
            return None, None


    # Finds by how much the rectangle overflows the image
    def __findOverflow(self, rectCoordinates, imageWidth, imageHeight):
        (x1, y1, x2, y2) = rectCoordinates
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
        for vc in self.__virtualCameras:
            vc.decreaseTimeToLive()
            if not vc.isAlive():
                self.__virtualCameras.remove(vc)

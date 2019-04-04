import math
import numpy as np

from PyQt5.QtCore import QTimer

from .virtual_camera import VirtualCamera


class VirtualCameraManager:

    def __init__(self, imgMinElevation, imgMaxElevation):
        self.__virtualCameras = []

        self.virtualCameraMinHeight = 0.3

        self.imgMinElevation = imgMinElevation
        self.imgMaxElevation = imgMaxElevation

        # Distance at which a new vc will be created instead of moved
        self.newVirtualCameraThreshold = 500

        # 3:4 (portrait)
        self.aspectRatio = 3 / 4

        # Change in position that cause a move of the virtual camera
        self.positionChangedThreshold = 0.005

        # Change in dimension that cause a resize of the virtual camera
        self.dimensionChangeThreshold = 0

        # Face scale factor to get the person's portrait (with shoulders)
        self.portraitScaleFactor = 4

        # Range in angles where we consider faces to be duplicate
        self.duplicateFaceAngleRange = 0.05

        # Garbage collector unused virtual cameras. Ticks every second
        self.timer = QTimer()
        self.timer.timeout.connect(self.__garbageCollect)
        self.timer.start(1000)


    # Updates virtual camera move and resize animations every frame based on time since last frame
    def update(self, frameTime):      
        for vc in self.__virtualCameras:
            distance = self.__distanceBetweenTwoSphericalAngles(vc.getMiddlePosition(), vc.positionGoal)
            if distance > self.positionChangedThreshold:    
                azimuth, elevation = vc.getMiddlePosition()
                goal = vc.positionGoal
                dista = self.__findSignedAzimuth360Distance(azimuth, goal[0])
                diste = goal[1] - elevation
                da = dista * frameTime
                de = diste * frameTime
                self.__tryMoveVirtualCamera(vc, ((azimuth + da), elevation + de))

            if abs(vc.sizeGoal[0] - vc.getAzimuthSpan())  > self.dimensionChangeThreshold and \
               abs(vc.sizeGoal[1] - vc.getElevationSpan()) > self.dimensionChangeThreshold:
                dh = vc.sizeGoal[1] - vc.getElevationSpan()
                resizeFactor = (vc.getElevationSpan() + dh * frameTime) / vc.getElevationSpan()
                self.__tryResizeVirtualCamera(vc, resizeFactor)


    # Updates the associated face (if any) for each virtual camera
    def updateFaces(self, faces):

        uniqueFaces = []
        facePositions = []

        # Look for duplicate faces and ignore them
        for face in faces:
            currentFacePosition = face.getMiddlePosition()

            faceAlreadyExist = False
            for facePosition in facePositions:
                if currentFacePosition[0] > facePosition[0] - self.duplicateFaceAngleRange \
                   and currentFacePosition[0] < facePosition[0] + self.duplicateFaceAngleRange \
                   and currentFacePosition[1] > facePosition[1] - self.duplicateFaceAngleRange \
                   and currentFacePosition[1] < facePosition[1] + self.duplicateFaceAngleRange :
                    faceAlreadyExist = True
                    break
                    
            if not faceAlreadyExist:
                facePositions.append(currentFacePosition)
                uniqueFaces.append(face)

        # Find matches between existing virtual cameras and detected faces
        matches = dict.fromkeys(self.__virtualCameras, (None, False))
        matches, unmatchedFaces = self.__tryFindMatches(matches, uniqueFaces)

        # Create new virtual cameras from faces that were not associated with a vc
        for unmatchedFace in unmatchedFaces:
            newVirtualCamera = VirtualCamera.createFromFace(unmatchedFace)
            ratio = max(self.portraitScaleFactor, self.virtualCameraMinHeight / newVirtualCamera.getElevationSpan())
            self.__tryResizeVirtualCamera(newVirtualCamera, ratio)
            newVirtualCamera.sizeGoal = (newVirtualCamera.getAzimuthSpan(), newVirtualCamera.getElevationSpan())
            self.__virtualCameras.append(newVirtualCamera)

        for vc, (face, processed) in matches.items():       
            # Found a face to associate the vc with, so we update the vc with its matched face
            if face:
                vc.resetTimeToLive()
                vc.face = face
                vc.positionGoal = face.getMiddlePosition()
                heightGoal = max(self.virtualCameraMinHeight, face.getElevationSpan() * self.portraitScaleFactor)
                vc.sizeGoal = (heightGoal * self.aspectRatio, heightGoal)


    # Returns a copy of the virtual cameras so the caller can't modify the original ones
    def getVirtualCameras(self):
        vcs = []
        for vc in self.__virtualCameras:
            vcs.append(VirtualCamera.copy(vc))
        return vcs


    def clear(self):
        self.__virtualCameras = []

     
    # Try to move the vc to the desired position. If a move in a certain dimension
    # makes the vc overflow the image, we disallow that move
    def __tryMoveVirtualCamera(self, virtualCamera, newPosition):
        (newAzimuth, newElevation) = newPosition
        elevationOverflow = self.__findElevationOverflow(newPosition)
        
        # There is an overflow, we need to remove it so the vc does not exit the image
        if elevationOverflow != 0:
            newElevation -= elevationOverflow

        virtualCamera.setMiddlePosition((newAzimuth, newElevation))


    # Try to resize the vc with the desired scale factor. 
    # If the new dimensions overflow the image, we move the image to remove the overflow
    def __tryResizeVirtualCamera(self, virtualCamera, scaleFactor):
        # Find the desired dimensions respecting the aspect ratio
        virtualCamera.setElevationSpan(min(virtualCamera.getElevationSpan() * scaleFactor,
                                           self.imgMaxElevation - self.imgMinElevation))
        virtualCamera.setAzimuthSpan(virtualCamera.getElevationSpan() * self.aspectRatio)

        # Remove caused overflow, if any
        self.__tryMoveVirtualCamera(virtualCamera, virtualCamera.getMiddlePosition())


    # Recursive method to find matches between virtual cameras and detected faces.
    # Returns vc-face matches plus all of the unmatched faces.
    def __tryFindMatches(self, matches, unmatchedFaces):

        # Returns if there is no unmatched vc's left or no unmatched faces left 
        unprocessed = {vc: (face, processed) for vc, (face, processed) in matches.items() if not processed}
        if len(unprocessed) < 1 or len(unmatchedFaces) < 1:
            return matches, unmatchedFaces

        # Map containing the closest vc to a face by distance
        faceDistanceVcMap = {}
        for vc, (face, processed) in unprocessed.items():  
            closestFaceIndex, distance = self.__findClosestFace(vc, unmatchedFaces)
            if closestFaceIndex is not None:
                currentFaceBestDistance, currentBestVc = faceDistanceVcMap.get(closestFaceIndex, (None, None))
                if currentFaceBestDistance is None or distance < currentFaceBestDistance:
                    faceDistanceVcMap[closestFaceIndex] = (distance, vc)
            
        # Update the matches with the closest face to the vc
        facesToMatch = unmatchedFaces
        for closestFaceIndex in faceDistanceVcMap.keys(): 
            (distance, vc) = faceDistanceVcMap[closestFaceIndex]
            closestFace = None
            if distance < self.newVirtualCameraThreshold:
                closestFace = facesToMatch[closestFaceIndex]
                unmatchedFaces = [face for face in unmatchedFaces if not face == closestFace]
            matches[vc] = (closestFace, True)

        # Call the method recursively to match the remaining unmatched vc's
        return self.__tryFindMatches(matches, unmatchedFaces)


    # Find the closest face to the specified virtual camera
    def __findClosestFace(self, virtualCamera, faces):
        closestIndex = None
        closestDistance = None
        for i in range(len(faces)):
            distance = self.__distanceBetweenTwoSphericalAngles(virtualCamera.getMiddlePosition(), faces[i].getMiddlePosition())
            if not closestDistance or distance < closestDistance:
                closestDistance = distance
                closestIndex = i

        if closestIndex is not None:
            return closestIndex, closestDistance
        else:
            return None, None


    # Finds by how much the rectangle overflows the image
    def __findElevationOverflow(self, position):
        azimuth, elevation = position
        elevationOverflow = 0

        if elevation < self.imgMinElevation:
            elevationOverflow = elevation
        elif elevation > self.imgMaxElevation:
            elevationOverflow = elevation - self.imgMaxElevation

        return elevationOverflow


    def __distanceBetweenTwoSphericalAngles(self, srcAngle, dstAngle):
        srcAngleAzimuth, srcAngleElevation = srcAngle
        dstAngleAzimuth, dstAngleElevation = dstAngle

        azimuthDistance = self.__findSignedAzimuth360Distance(srcAngleAzimuth, dstAngleAzimuth)
        elevationDistance = srcAngleElevation - dstAngleElevation

        return math.sqrt(azimuthDistance ** 2 + elevationDistance ** 2)


    def __findSignedAzimuth360Distance(self, srcAzimuth, dstAzimuth):
        distance = dstAzimuth - srcAzimuth
        absDistance = abs(distance)
        if absDistance > np.pi:
            absDistance = np.pi * 2 - absDistance
            
        if (distance + np.pi * 2) % (np.pi * 2) < np.pi:
            return absDistance
        else:
            return -absDistance


    # Removes the virtual cameras that were not associated with a face for some time
    def __garbageCollect(self):
        for vc in self.__virtualCameras:
            vc.decreaseTimeToLive()
            if not vc.isAlive():
                self.__virtualCameras.remove(vc)

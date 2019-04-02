from math import radians
from src.utils.spherical_angles_converter import SphericalAnglesConverter

class SourceClassifier():

    def __init__(self, cameraParams, rangeThreshold):
        self.rangeThreshold = radians(rangeThreshold)
        self.cameraParams = cameraParams
        self.__humanSources = {}


    def classifySources(self, virtualCameras, soundSources):
        for index, source in enumerate(soundSources):
            # there is no sound detected if the elevation is less than 0
            if source['elevation'] > 0:
                for virtualCamera in virtualCameras:
                    xFace, yFace = virtualCamera.face.getPosition()
                    azimuthFace, elevationFace = SphericalAnglesConverter.getSphericalAnglesFromImage(xFace, yFace, self.cameraParams)

                    if self.__isInRange(azimuthFace, source['azimuth']) and self.__isInRange(elevationFace, source['elevation']):
                            self.__humanSources[index] = source


    def getHumanSources(self):
        return self.__humanSources


    def __isInRange(self, faceAngle, soundAngle):
        maxDetectionAngle = soundAngle + self.rangeThreshold
        minDetectionAngle = soundAngle - self.rangeThreshold

        return faceAngle <= maxDetectionAngle and faceAngle >= minDetectionAngle


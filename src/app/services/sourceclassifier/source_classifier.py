from math import radians
from src.utils.spherical_angles_converter import SphericalAnglesConverter

class SourceClassifier():

    def __init__(self, cameraParams, rangeThreshold):
        self.rangeThreshold = radians(rangeThreshold)
        self.cameraParams = cameraParams

    def classifySources(self, virtualCameras, soundSources):
        humanSources = []
        
        for source in soundSources:
            # there is no sound detected if the elevation is less than 0
            if source['elevation'] > 0:
                for virtualCamera in virtualCameras:
                    xFace, yFace = virtualCamera.face.getPosition()
                    azimuthFace, elevationFace = SphericalAnglesConverter.getSphericalAnglesFromImage(xFace, yFace, self.cameraParams)

                    if self.__inRange(azimuthFace, source['azimuth']):
                        if self.__inRange(elevationFace, source['elevation']):
                            humanSources.append(source)

        print('human sources: ' + str(len(humanSources)))
        self.humanSources = humanSources

    def getHumanSources(self):
        return self.humanSources

    def __inRange(self, faceAngle, soundAngle):
        maxDetectionAngle = soundAngle + self.rangeThreshold
        minDetectionAngle = soundAngle - self.rangeThreshold

        if faceAngle <= maxDetectionAngle and faceAngle >= minDetectionAngle:
            return True

            
import numpy as np
from math import radians
from src.utils.spherical_angles_converter import SphericalAnglesConverter

class SourceClassifier():

    def __init__(self, rangeThreshold):
        self.rangeThreshold = radians(rangeThreshold)
        self.humanSources = {}


    def classifySources(self, virtualCameras, soundSources):
        for index, source in enumerate(soundSources):
            # there is no sound detected if the elevation is less than 0
            if source['elevation'] > 0:
                for virtualCamera in virtualCameras:

                    azimuth, elevation = virtualCamera.face.getMiddlePosition()
                    
                    # Because camera angles are clockwise and odas not
                    azimuth = (2 * np.pi) - azimuth

                    if self.__isInRange(azimuth, source['azimuth']) and self.__isInRange(elevation, source['elevation']):
                            self.humanSources[index] = source


    def __isInRange(self, faceAngle, soundAngle):
        maxDetectionAngle = soundAngle + self.rangeThreshold
        minDetectionAngle = soundAngle - self.rangeThreshold

        return faceAngle <= maxDetectionAngle and faceAngle >= minDetectionAngle


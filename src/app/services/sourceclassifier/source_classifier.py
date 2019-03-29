from src.utils.spherical_angles_converter import SphericalAnglesConverter
import math
import cv2

class SourceClassifier():

    def __init__(self, cameraParams, rangeThreshold):
        self.cameraParams = cameraParams  

        # acceptance threshold in degrees
        self.rangeThreshold = rangeThreshold                          

    def classifySources(self, virtualCameras, soundSources):
        humanSources = []
        noiseSources = []
        for index in soundSources:
            if soundSources[index]['elevation'] > 0:
                for virtualCamera in virtualCameras:
                    face = {}
                    face['x'], face['y'] = virtualCamera.face.getPosition()
                    faceAngles = SphericalAnglesConverter.getSphericalAnglesFromImage(face['x'], face['y'],
                                                                                      math.radians(self.cameraParams['fisheyeAngle']),
                                                                                      self.cameraParams['baseDonutSlice'],
                                                                                      self.cameraParams['dewarpingParameters'])

                    face['azimuth'] = math.degrees(faceAngles[0])
                    face['elevation'] = math.degrees(faceAngles[1])
                    
                    if self.__inRange(face['azimuth'], soundSources[index]['azimuth']):
                        if self.__inRange(face['elevation'], soundSources[index]['elevation']):
                            humanSources.append(soundSources[index])
                        else:
                            noiseSources.append(soundSources[index])
                    else:
                        noiseSources.append(soundSources[index])

        print('human sources: ' + str(len(humanSources)) + ', non-human sources: ' + str(len(noiseSources)))
        self.humanSources = humanSources
        self.noiseSources = noiseSources

    def getHumanSources(self):
        return self.humanSources
    
    def getNoiseSources(self):
        return self.noiseSources

    def __inRange(self, faceAngle, soundAngle):
        if faceAngle <= (soundAngle + self.rangeThreshold) and faceAngle >= (soundAngle - self.rangeThreshold):
            return True

            
from src.utils.spherical_angles_converter import SphericalAnglesConverter
import math
import cv2

class SourceClassifier():

    def __init__(self, cameraParams):
        self.cameraParams = cameraParams

    def drawSoundSources(self, image, virtualCameras, soundSources):
        for virtualCamera in virtualCameras:
            face = []
            face['x'], face['y'] = virtualCamera.getPosition()
            face['azimuth'], face['elevation'] = SphericalAnglesConverter.getSphericalAnglesFromImage(face['xPos'], 
                                                                                                      face['yPos'],
                                                                                                      self.cameraParams.fisheyeAngle,
                                                                                                      self.cameraParams.baseDonutSlice,
                                                                                                      self.cameraParams.dewarpingParameters)
            
            source = getSoundSourceNearLocation(face['azimuth'], face['elevation'], soundSources, 30)
            if(source):
                xSoundSource = source['azimuth'] * face['x'] / face['azimuth']
                ySoundSource = source['elevation'] * face['y'] / face['elevation']
                cv2.circle(image, (xSoundSource, ySoundSource), 50, (0,0,255), 3)
            

    def getSoundSourceNearLocation(azimuth, elevation, soundSources, threshold):
        for index in soundSources:
            if soundSources[index]['azimuth'] <= azimuth + threshold and soundSources[index]['azimuth'] >= azimuth - threshold:
                if soundSources[index]['elevation'] <= elevation + threshold and soundSources[index]['elevation'] >= elevation - threshold:
                    return soundSources[index]

                

    def classifySources(self, faces, soundSources, threshold):
        threshold = math.radians(threshold)
        humanSources = []
        for source in soundSources:
            for face in faces:
                (x1, y1, x2, y2) = face
                xFace, yFace = x1, y1
                faceAngle = {}
                faceAngle['azimuth'], faceAngle['elevation'] = SphericalAnglesConverter.getSphericalAnglesFromImage(xFace, yFace, self.cameraParams)

                if soundSources[source]['azimuth'] <= faceAngle['azimuth'] + threshold and soundSources[source]['azimuth'] >= faceAngle['azimuth'] - threshold:
                    if soundSources[source]['elevation'] <= faceAngle['elevation'] + threshold and soundSources[source]['elevation'] >= faceAngle['elevation'] - threshold:
                        print('sound' + str(soundSources[source]['azimuth']) + ', ' + str(soundSources[source]['elevation']))
                        print('face' + str(faceAngle['azimuth']) + ', ' + str(faceAngle['elevation']))
                        # sound source is near a face, the sound is most likely human speech
                        humanSources.append(source)

        return humanSources

    def classifyFaces(self, faces, soundSources, threshold):
        threshold = math.radians(threshold)
        speakers = []
        for face in faces:
            for source in soundSources:
                (x1, y1, x2, y2) = face
                xFace, yFace = x1, y1
                faceAngle = {}
                faceAngle['azimuth'], faceAngle['elevation'] = SphericalAnglesConverter.getSphericalAnglesFromImage(xFace, yFace, self.cameraParams)
                print('face: ' + str(faceAngle['azimuth']) + ', ' + str(faceAngle['elevation']))
                print('sound: ' + str(soundSources[source]['azimuth']) + ', ' + str(soundSources[source]['elevation']))
                if faceAngle['azimuth'] <= (soundSources[source]['azimuth'] + threshold) and faceAngle['azimuth'] >= (soundSources[source]['azimuth'] - threshold):
                    if faceAngle['elevation'] <= (soundSources[source]['elevation'] + threshold) and faceAngle['elevation'] >= (soundSources[source]['elevation'] - threshold):
                        # sound source is near a face, the human is speaking
                        print('YOU ARE A HUMAN')
                        speakers.append(face)

        return speakers

            
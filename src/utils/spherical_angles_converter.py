import math

import numpy as np

from src.utils.dewarping_helper import DewarpingHelper


# All angles are in radian
class SphericalAnglesConverter:

    def __init__(self):
        pass


    @staticmethod
    def getAzimuthFromPosition(x, y):
        return math.atan2(y, x) % (2 * math.pi)


    @staticmethod
    def getElevationFromPosition(x, y, z):
        xyHypotenuse = math.sqrt(y**2 + x**2)
        return math.atan2(z, xyHypotenuse) % (2 * math.pi)


    @staticmethod
    def getSphericalAnglesFromImage(xPixel, yPixel, cameraParams, isClockwise = False):
        fisheyeAngle = cameraParams['fisheyeAngle']
        baseDonutSlice = cameraParams['baseDonutSlice']
        dewarpingParameters = cameraParams['dewarpingParameters']
        
        xSourcePixel, ySourcePixel = DewarpingHelper.getSourcePixelFromDewarpedImage(xPixel, yPixel, dewarpingParameters)

        if fisheyeAngle > 2 * np.pi:
            raise Exception("Fisheye angle must be in radian!")

        dx = xSourcePixel - baseDonutSlice.xCenter
        dy = ySourcePixel - baseDonutSlice.yCenter
        distanceFromCenter = np.sqrt(dx**2 + dy**2)
        distanceFromBorder = baseDonutSlice.xCenter - distanceFromCenter
        ratio = (distanceFromBorder / baseDonutSlice.xCenter)

        elevation = ratio * (fisheyeAngle / 2) + (np.pi / 2 - (fisheyeAngle / 2))
        azimuth = 0

        if dx >= 0 and dy > 0:
            azimuth = np.arctan(dx / dy)
        elif dx <= 0 and dy < 0:
            azimuth = np.arctan(-dx / -dy) + np.pi
        elif dx > 0 and dy <= 0:
            azimuth = np.arctan(-dy / dx) + np.pi / 2
        elif dx < 0 and dy >= 0:
            azimuth = np.arctan(dy / -dx) + 3 * np.pi / 2

        if isClockwise == False:
            azimuth = (2 * np.pi) - azimuth

        return azimuth, elevation


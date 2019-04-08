import numpy as np

from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import DewarpingParameters
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import DonutSlice


class DewarpingHelper:

    def __init__(self):
        pass


    @staticmethod
    def getDewarpingParameters(baseDonutSlice, topDistorsionFactor, bottomDistorsionFactor):
        # Distance between center of baseDonutSlice and newDonutSlice to calculate
        centersDistance = topDistorsionFactor * 10000

        # Return a new donut mapping based on the one passed, which have properties to reduce the distorsion in the image
        newDonutSlice = DewarpingHelper.createDewarpingDonutSlice(baseDonutSlice, centersDistance)

        return DewarpingHelper.getDewarpingParametersFromNewDonutSlice(baseDonutSlice, newDonutSlice, centersDistance, bottomDistorsionFactor)

    
    @staticmethod
    def getDewarpingParametersFromNewDonutSlice(baseDonutSlice, newDonutSlice, centersDistance, bottomDistorsionFactor):
        # Radius of circle which would be in the middle of the dewarped image if no radius factor was applied to dewarping
        centerRadius = (newDonutSlice.inRadius + newDonutSlice.outRadius) / 2

        # Offset in x in order for the mapping to be in the right section of the source image (changes the angle of mapping)
        xOffset = (newDonutSlice.middleAngle - newDonutSlice.angleSpan / 2) * centerRadius

        # Difference between the outside radius of the base donut and the new one
        outRadiusDiff = baseDonutSlice.outRadius + centersDistance - newDonutSlice.outRadius 

        # Width and Height of the dewarped image
        dewarpHeight = newDonutSlice.outRadius - newDonutSlice.inRadius
        dewarpWidth = newDonutSlice.angleSpan * centerRadius

        return DewarpingParameters(newDonutSlice.xCenter, newDonutSlice.yCenter, dewarpWidth, dewarpHeight, \
            newDonutSlice.inRadius, centerRadius, outRadiusDiff, xOffset, bottomDistorsionFactor)


    @staticmethod
    def getVirtualCameraDewarpingParameters(rect, baseDonutSlice, dewarpingParameters, topDistorsionFactor):
        (x1, y1, x2, y2) = rect
        centersDistance = topDistorsionFactor * 10000

        newDonutSlice = DewarpingHelper.createDewarpingDonutSlice(baseDonutSlice, centersDistance)

        print('(x1, y1, x2, y2) = ', x1, ' ', y1, ' ', x2, ' ', y2)
        print('newDonutSlice.angleSpan', newDonutSlice.angleSpan)
        print('newDonutSlice.middleAngle', newDonutSlice.middleAngle)

        vcWidth = x2 - x1
        angleSpan = (newDonutSlice.angleSpan * vcWidth) / dewarpingParameters.dewarpWidth
        newDonutSlice.angleSpan = angleSpan

        middleVcX = (vcWidth) / 2 + x1
        print('middleVcX', middleVcX)
        dx = dewarpingParameters.dewarpWidth / 2 - middleVcX
        print('dx', dx)
        newDonutSlice.middleAngle = newDonutSlice.middleAngle - (newDonutSlice.angleSpan * dx) / dewarpingParameters.dewarpWidth
        
        inRadius = DewarpingHelper.getPixelDewarpedParameterRadius(x1, y1, dewarpingParameters)
        outRadius = DewarpingHelper.getPixelDewarpedParameterRadius(x1, y2, dewarpingParameters)

        print('inRadius', inRadius)
        print('outRadius', outRadius)

        newDewarpingParameters = DewarpingHelper.getDewarpingParametersFromNewDonutSlice(baseDonutSlice, \
            newDonutSlice, centersDistance, dewarpingParameters.bottomDistorsionFactor)
        newDewarpingParameters.topOffset = inRadius - dewarpingParameters.inRadius
        newDewarpingParameters.bottomOffset = newDonutSlice.outRadius - outRadius

        

        print('topOffset', newDewarpingParameters.topOffset)
        print('bottomOffset', newDewarpingParameters.bottomOffset)
        print('angleSpan', newDonutSlice.angleSpan)
        print('middleAngle', newDonutSlice.middleAngle)
	   
        return newDewarpingParameters


    @staticmethod
    def getPixelDewarpedParameterRadius(x, y, dewarpingParameters):
        xSourcePixel, ySourcePixel = DewarpingHelper.getSourcePixelFromDewarpedImage(x, y, dewarpingParameters)
        dx = dewarpingParameters.xCenter - xSourcePixel
        dy = dewarpingParameters.yCenter - ySourcePixel
        return np.sqrt(dx**2 + dy**2)


    @staticmethod
    def createDewarpingDonutSlice(baseDonutSlice, centersDistance):
        if baseDonutSlice.middleAngle > 2 * np.pi or baseDonutSlice.angleSpan > 2 * np.pi:
            raise Exception("Donut slice angles must be in radian!")

        xNewCenter = baseDonutSlice.xCenter - np.sin(baseDonutSlice.middleAngle) * centersDistance
        yNewCenter = baseDonutSlice.yCenter - np.cos(baseDonutSlice.middleAngle) * centersDistance

        # Length of a line which start from the new center, pass by image center and end to form the side of a right-angled 
        # triangle which other point is on the circle of center (xImageCenter, yImageCenter) and radius (outRadius)
        d = np.cos(baseDonutSlice.angleSpan / 2) * baseDonutSlice.outRadius + centersDistance

        newInRadius = baseDonutSlice.inRadius + centersDistance
        newOutRadius = np.sqrt(d**2 + (np.sin(baseDonutSlice.angleSpan / 2) * baseDonutSlice.outRadius)**2)
        newAngleSpan = np.arccos(d / newOutRadius) * 2

        return DonutSlice(xNewCenter, yNewCenter, newInRadius, newOutRadius, baseDonutSlice.middleAngle, newAngleSpan)


    @staticmethod
    def getSourcePixelFromDewarpedImage(xPixel, yPixel, dewarpingParameters):
        xRadiusFactor = np.sin((np.pi * xPixel) / dewarpingParameters.dewarpWidth)
        yRadiusFactor = np.sin((np.pi * yPixel) / (dewarpingParameters.dewarpHeight * 2))

        r = yPixel + dewarpingParameters.inRadius + dewarpingParameters.outRadiusDiff * \
            xRadiusFactor * yRadiusFactor * (1 - dewarpingParameters.bottomDistorsionFactor)
        theta = (xPixel + dewarpingParameters.xOffset) / dewarpingParameters.centerRadius
        xSourcePixel = dewarpingParameters.xCenter + r * np.sin(theta)
        ySourcePixel = dewarpingParameters.yCenter + r * np.cos(theta)

        return xSourcePixel, ySourcePixel
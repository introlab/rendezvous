import cv2
import numpy as np
import time
from collections import namedtuple

DonutMapping = namedtuple('DonutMapping', 'xCenter yCenter inRadius outRadius middleAngle angleSpan')
DebugImageInfoParam = namedtuple('DebugImageInfoParam', 'donutMapping newDonutMapping center \
    newCenter bottomLeft bottomRight centerRadius')


class VideoStream:

    def __init__(self, cameraPort, imageWidth, imageHeight, fps, fourcc):
        self.camera = cv2.VideoCapture(cameraPort)
        self.cameraPort = cameraPort
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.fps = fps
        self.fourcc = fourcc

    # Displays the source and dewarped image, set debug to true to show the areas of the calculations
    def startStream(self, donutMapping, centersDistance, debug):
        self.__initalizeCamera()

        # Changing the codec of the camera throws an exception on camera.read every two execution for whatever reason
        try:
            self.camera.read()
        except:
            self.camera.release()
            self.camera.open(self.cameraPort)
            self.__initalizeCamera()

        self.printCameraSettings()

        try:
            xMap, yMap = self.__readMaps(donutMapping, centersDistance)
        except:
            self.buildMaps(donutMapping, centersDistance)
            xMap, yMap = self.__readMaps(donutMapping, centersDistance)

        if debug:
            print('DEBUG enabled')
            debugImageInfoParam = self.__createDebugImageInfoParam(donutMapping, centersDistance)

        print('Press ESCAPE key to exit')

        while(True):
            success, frame = self.camera.read()

            if success:
                if debug:
                    self.__addDebugInfoToImage(frame, debugImageInfoParam)

                cv2.imshow('frame', cv2.resize(frame, (720, 540)))

                dewarpedFrame = self.__unwarp(frame, xMap, yMap)
                cv2.imshow('unwarp', cv2.resize(dewarpedFrame, (775, 452)))

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.camera.release()
        cv2.destroyAllWindows()

    
    # Build maps for the unwarping of fisheye image
    def buildMaps(self, baseDonutMapping, centersDistance):
        print("Building Maps...")

        # Return a new donut mapping based on the one passed, which have properties to reduce the distorsion in the image
        newDonutMapping = self.__createDewarpingDonutMapping(baseDonutMapping, centersDistance)

        # Radius of circle which would be in the middle of the dewarped image if no radius factor was applied to dewarping
        centerRadius = (newDonutMapping.inRadius + newDonutMapping.outRadius) / 2

        # Offset in x in order for the mapping to be in the right section of the source image (changes the angle of mapping)
        xOffset = (newDonutMapping.middleAngle - newDonutMapping.angleSpan / 2) * centerRadius

        # Difference between the outside radius of the base donut and the new one
        outRadiusDiff = baseDonutMapping.outRadius + centersDistance - newDonutMapping.outRadius 

        # Width and Height of the dewarped image
        dewarpHeight = newDonutMapping.outRadius - newDonutMapping.inRadius
        dewarpWidth = newDonutMapping.angleSpan * centerRadius

        # Build maps to ajust the radius used when mapping the pixels of the dewarped image to pixels of the source image
        xRadiusFactorMap = np.zeros(int(dewarpWidth), np.float32)
        for x in range(0, int(dewarpWidth) - 1):
            xRadiusFactorMap.itemset(x, np.sin((np.pi * x) / dewarpWidth))
        
        yRadiusFactorMap = np.zeros(int(dewarpHeight), np.float32)
        for y in range(0, int(dewarpHeight - 1)):
            yRadiusFactorMap.itemset(y, np.sin((np.pi * y) / (dewarpHeight * 2)))

        # Build the pixel coordinate maps used to generate the dewarped image
        xMap = np.zeros((int(dewarpHeight), int(dewarpWidth)), np.float32)
        yMap = np.zeros((int(dewarpHeight), int(dewarpWidth)), np.float32)
        for y in range(0, int(dewarpHeight - 1)):
            for x in range(0, int(dewarpWidth) - 1): 
                r = y + newDonutMapping.inRadius + outRadiusDiff * xRadiusFactorMap[x] * yRadiusFactorMap[y]
                theta = (x + xOffset) / centerRadius
                xS = newDonutMapping.xCenter + r * np.sin(theta)
                yS = newDonutMapping.yCenter + r * np.cos(theta)
                xMap.itemset((y, x), int(xS))
                yMap.itemset((y, x), int(yS))

        self.__saveMaps(baseDonutMapping, centersDistance, xMap, yMap)


    def printCameraSettings(self):
        print('Image width = {width}'.format(width=self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print('Image height = {height}'.format(height=self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print('Codec = {codec}'.format(codec=self.__decode_fourcc(self.camera.get(cv2.CAP_PROP_FOURCC))))
        print('FPS = {fps}'.format(fps=self.camera.get(cv2.CAP_PROP_FPS)))


    def __initalizeCamera(self):
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(self.fourcc[0], self.fourcc[1], self.fourcc[2], self.fourcc[3]))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.imageWidth)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.imageHeight)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)


    # Unwarp video stream from fisheye to pano
    def __unwarp(self, img, xmap, ymap):
        output = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
        return output


    # Create the dataset required to display the debug lines on the source image
    def __createDebugImageInfoParam(self, donutMapping, centersDistance):
        newDonutMapping = self.__createDewarpingDonutMapping(donutMapping, centersDistance)
        center = (int(donutMapping.xCenter), int(donutMapping.yCenter))
        newCenter = (int(newDonutMapping.xCenter), int(newDonutMapping.yCenter))
        theta0 = donutMapping.middleAngle - donutMapping.angleSpan / 2
        theta1 = donutMapping.middleAngle + donutMapping.angleSpan / 2
        bottomLeft = (int(donutMapping.xCenter + np.sin(theta0) * donutMapping.outRadius), \
            int(donutMapping.yCenter + np.cos(theta0) * donutMapping.outRadius))
        bottomRight = (int(donutMapping.xCenter + np.sin(theta1) * donutMapping.outRadius), \
            int(donutMapping.yCenter + np.cos(theta1) * donutMapping.outRadius))
        centerRadius = (newDonutMapping.inRadius + newDonutMapping.outRadius) / 2

        return DebugImageInfoParam(donutMapping=donutMapping, newDonutMapping=newDonutMapping, center=center, \
            newCenter=newCenter, bottomLeft=bottomLeft, bottomRight=bottomRight, centerRadius=centerRadius)


    # Add debug lines on the source image
    def __addDebugInfoToImage(self, frame, debugImageInfoParam):
        donutMapping = debugImageInfoParam.donutMapping
        newDonutMapping = debugImageInfoParam.newDonutMapping

        cv2.circle(frame, debugImageInfoParam.center, donutMapping.inRadius, (255,0,255), 5)
        cv2.circle(frame, debugImageInfoParam.center, donutMapping.outRadius, (255,0,255), 5)
        cv2.circle(frame, debugImageInfoParam.center, int((donutMapping.inRadius + donutMapping.outRadius) / 2), (255,255,0), 5)

        cv2.line(frame, debugImageInfoParam.center, debugImageInfoParam.bottomLeft, (255,0,255), 5)
        cv2.line(frame, debugImageInfoParam.center, debugImageInfoParam.bottomRight, (255,0,255), 5)

        cv2.circle(frame, debugImageInfoParam.newCenter, int(newDonutMapping.inRadius), (255,0,122), 5)
        cv2.circle(frame, debugImageInfoParam.newCenter, int(newDonutMapping.outRadius), (255,0,122), 5)
        cv2.circle(frame, debugImageInfoParam.newCenter, int(debugImageInfoParam.centerRadius), (122,0,255), 5)

        cv2.line(frame, debugImageInfoParam.newCenter, (int(newDonutMapping.xCenter + np.sin(donutMapping.middleAngle) * newDonutMapping.outRadius), \
            int(newDonutMapping.yCenter + np.cos(donutMapping.middleAngle) * newDonutMapping.outRadius)), (255,0,122), 5)
        cv2.line(frame, debugImageInfoParam.newCenter, debugImageInfoParam.bottomLeft, (122,0,255), 5)
        cv2.line(frame, debugImageInfoParam.newCenter, debugImageInfoParam.bottomRight, (122,0,255), 5)


    # Return 4 chars reprenting codec
    def __decode_fourcc(self, cc):
        return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


    # It is called a donut mapping because the pixels in the dewarped image will be mapped to 
    # pixels from the camera image in between two circles (of radius inRadius and outRadius)
    def __createDewarpingDonutMapping(self, baseDonutMapping, centersDistance):
        xNewCenter = baseDonutMapping.xCenter - np.sin(baseDonutMapping.middleAngle) * centersDistance
        yNewCenter = baseDonutMapping.yCenter - np.cos(baseDonutMapping.middleAngle) * centersDistance

        # Length of a line which start from the new center, pass by image center and end to form the side of a right-angled 
        # triangle which other point is on the circle of center (xImageCenter, yImageCenter) and radius (outRadius)
        d = np.cos(baseDonutMapping.angleSpan / 2) * baseDonutMapping.outRadius + centersDistance

        newInRadius = baseDonutMapping.inRadius + centersDistance
        newOutRadius = np.sqrt(d**2 + (np.sin(baseDonutMapping.angleSpan / 2) * baseDonutMapping.outRadius)**2)
        newAngleSpan = np.arccos(d / newOutRadius) * 2

        return DonutMapping(xCenter=xNewCenter, yCenter=yNewCenter, inRadius = newInRadius, \
            outRadius=newOutRadius, middleAngle=baseDonutMapping.middleAngle, angleSpan=newAngleSpan)


    def __saveMaps(self, donutMapping, centersDistance, xMap, yMap):
        np.save('../../config/maps/{xC}-{yC}-{iR}-{oR}-{mA}-{sA}-{cD}-xmap.npy' \
            .format(xC=int(donutMapping.xCenter), yC=int(donutMapping.yCenter), iR=int(donutMapping.inRadius), \
            oR=int(donutMapping.outRadius), mA=int(np.rad2deg(donutMapping.middleAngle)), \
            sA=int(np.rad2deg(donutMapping.angleSpan)), cD=int(centersDistance)), xMap)

        np.save('../../config/maps/{xC}-{yC}-{iR}-{oR}-{mA}-{sA}-{cD}-ymap.npy' \
            .format(xC=int(donutMapping.xCenter), yC=int(donutMapping.yCenter), iR=int(donutMapping.inRadius), \
            oR=int(donutMapping.outRadius), mA=int(np.rad2deg(donutMapping.middleAngle)), \
            sA=int(np.rad2deg(donutMapping.angleSpan)), cD=int(centersDistance)), yMap)


    def __readMaps(self, donutMapping, centersDistance):
        xMap = np.load('../../config/maps/{xC}-{yC}-{iR}-{oR}-{mA}-{sA}-{cD}-xmap.npy' \
            .format(xC=int(donutMapping.xCenter), yC=int(donutMapping.yCenter), iR=int(donutMapping.inRadius), \
            oR=int(donutMapping.outRadius), mA=int(np.rad2deg(donutMapping.middleAngle)), \
            sA=int(np.rad2deg(donutMapping.angleSpan)), cD=int(centersDistance)))

        yMap = np.load('../../config/maps/{xC}-{yC}-{iR}-{oR}-{mA}-{sA}-{cD}-ymap.npy' \
            .format(xC=int(donutMapping.xCenter), yC=int(donutMapping.yCenter), iR=int(donutMapping.inRadius), \
            oR=int(donutMapping.outRadius), mA=int(np.rad2deg(donutMapping.middleAngle)), \
            sA=int(np.rad2deg(donutMapping.angleSpan)), cD=int(centersDistance)))

        return xMap, yMap
    

import time
import os
from collections import namedtuple

import cv2
import numpy as np
from pathlib import Path

DonutSlice = namedtuple('DonutSlice', 'xCenter yCenter inRadius outRadius middleAngle angleSpan')
DebugImageInfoParam = namedtuple('DebugImageInfoParam', 'donutSlice newDonutSlice center \
    newCenter bottomLeft bottomRight centerRadius')

rootDirectory = os.path.realpath(Path(__file__).parents[3])


class VideoStream:

    def __init__(self, cameraPort, imageWidth, imageHeight, fps, fourcc):
        self.camera = cv2.VideoCapture(cameraPort)
        self.cameraPort = cameraPort
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.fps = fps
        self.fourcc = fourcc


    # Displays the source and dewarped image, set debug to true to show the areas of the calculations
    def startStream(self, donutSlice, topDistorsionFactor, bottomDistorsionFactor, debug):
        self.__initalizeCamera()

        # Changing the codec of the camera throws an exception on camera.read every two execution for whatever reason
        try:
            self.camera.read()
        except:
            self.camera.release()
            self.camera.open(self.cameraPort)
            self.__initalizeCamera()

        self.printCameraSettings()

        # If the maps doesn't exist, create them
        try:
            xMap, yMap = self.__readMaps(donutSlice, topDistorsionFactor, bottomDistorsionFactor)
        except:
            self.buildMaps(donutSlice, topDistorsionFactor, bottomDistorsionFactor)
            xMap, yMap = self.__readMaps(donutSlice, topDistorsionFactor, bottomDistorsionFactor)

        if debug:
            print('DEBUG enabled')
            debugImageInfoParam = self.__createDebugImageInfoParam(donutSlice, topDistorsionFactor)

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
    def buildMaps(self, baseDonutSlice, topDistorsionFactor, bottomDistorsionFactor):
        print("Building Maps...")

        # Distance between center of baseDonutSlice and newDonutSlice to calculate
        centersDistance = topDistorsionFactor * 10000

        # Return a new donut mapping based on the one passed, which have properties to reduce the distorsion in the image
        newDonutSlice = self.__createDewarpingDonutSlice(baseDonutSlice, centersDistance)

        # Radius of circle which would be in the middle of the dewarped image if no radius factor was applied to dewarping
        centerRadius = (newDonutSlice.inRadius + newDonutSlice.outRadius) / 2

        # Offset in x in order for the mapping to be in the right section of the source image (changes the angle of mapping)
        xOffset = (newDonutSlice.middleAngle - newDonutSlice.angleSpan / 2) * centerRadius

        # Difference between the outside radius of the base donut and the new one
        outRadiusDiff = baseDonutSlice.outRadius + centersDistance - newDonutSlice.outRadius 

        # Width and Height of the dewarped image
        dewarpHeight = newDonutSlice.outRadius - newDonutSlice.inRadius
        dewarpWidth = newDonutSlice.angleSpan * centerRadius

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
                r = y + newDonutSlice.inRadius + outRadiusDiff * xRadiusFactorMap[x] \
                    * yRadiusFactorMap[y] * (1 - bottomDistorsionFactor)
                theta = (x + xOffset) / centerRadius
                xS = newDonutSlice.xCenter + r * np.sin(theta)
                yS = newDonutSlice.yCenter + r * np.cos(theta)
                xMap.itemset((y, x), int(xS))
                yMap.itemset((y, x), int(yS))

        self.__saveMaps(baseDonutSlice, topDistorsionFactor, bottomDistorsionFactor, xMap, yMap)


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
    def __createDebugImageInfoParam(self, donutSlice, topDistorsionFactor):
        centersDistance = topDistorsionFactor * 10000
        newDonutSlice = self.__createDewarpingDonutSlice(donutSlice, centersDistance)
        center = (int(donutSlice.xCenter), int(donutSlice.yCenter))
        newCenter = (int(newDonutSlice.xCenter), int(newDonutSlice.yCenter))
        theta0 = donutSlice.middleAngle - donutSlice.angleSpan / 2
        theta1 = donutSlice.middleAngle + donutSlice.angleSpan / 2
        bottomLeft = (int(donutSlice.xCenter + np.sin(theta0) * donutSlice.outRadius), \
            int(donutSlice.yCenter + np.cos(theta0) * donutSlice.outRadius))
        bottomRight = (int(donutSlice.xCenter + np.sin(theta1) * donutSlice.outRadius), \
            int(donutSlice.yCenter + np.cos(theta1) * donutSlice.outRadius))
        centerRadius = (newDonutSlice.inRadius + newDonutSlice.outRadius) / 2

        return DebugImageInfoParam(donutSlice=donutSlice, newDonutSlice=newDonutSlice, center=center, \
            newCenter=newCenter, bottomLeft=bottomLeft, bottomRight=bottomRight, centerRadius=centerRadius)


    # Add debug lines on the source image
    def __addDebugInfoToImage(self, frame, debugImageInfoParam):
        donutSlice = debugImageInfoParam.donutSlice
        newDonutSlice = debugImageInfoParam.newDonutSlice

        cv2.circle(frame, debugImageInfoParam.center, donutSlice.inRadius, (255,0,255), 5)
        cv2.circle(frame, debugImageInfoParam.center, donutSlice.outRadius, (255,0,255), 5)
        cv2.circle(frame, debugImageInfoParam.center, int((donutSlice.inRadius + donutSlice.outRadius) / 2), (255,255,0), 5)

        cv2.line(frame, debugImageInfoParam.center, debugImageInfoParam.bottomLeft, (255,0,255), 5)
        cv2.line(frame, debugImageInfoParam.center, debugImageInfoParam.bottomRight, (255,0,255), 5)

        cv2.circle(frame, debugImageInfoParam.newCenter, int(newDonutSlice.inRadius), (255,0,122), 5)
        cv2.circle(frame, debugImageInfoParam.newCenter, int(newDonutSlice.outRadius), (255,0,122), 5)
        cv2.circle(frame, debugImageInfoParam.newCenter, int(debugImageInfoParam.centerRadius), (122,0,255), 5)

        cv2.line(frame, debugImageInfoParam.newCenter, (int(newDonutSlice.xCenter + np.sin(donutSlice.middleAngle) * newDonutSlice.outRadius), \
            int(newDonutSlice.yCenter + np.cos(donutSlice.middleAngle) * newDonutSlice.outRadius)), (255,0,122), 5)
        cv2.line(frame, debugImageInfoParam.newCenter, debugImageInfoParam.bottomLeft, (122,0,255), 5)
        cv2.line(frame, debugImageInfoParam.newCenter, debugImageInfoParam.bottomRight, (122,0,255), 5)


    # Return 4 chars reprenting codec
    def __decode_fourcc(self, cc):
        return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


    # It is called a donut slice because the pixels in the dewarped image will be mapped to 
    # pixels from the camera image in between two circles (of radius inRadius and outRadius) and for a given angle
    def __createDewarpingDonutSlice(self, baseDonutSlice, centersDistance):
        xNewCenter = baseDonutSlice.xCenter - np.sin(baseDonutSlice.middleAngle) * centersDistance
        yNewCenter = baseDonutSlice.yCenter - np.cos(baseDonutSlice.middleAngle) * centersDistance

        # Length of a line which start from the new center, pass by image center and end to form the side of a right-angled 
        # triangle which other point is on the circle of center (xImageCenter, yImageCenter) and radius (outRadius)
        d = np.cos(baseDonutSlice.angleSpan / 2) * baseDonutSlice.outRadius + centersDistance

        newInRadius = baseDonutSlice.inRadius + centersDistance
        newOutRadius = np.sqrt(d**2 + (np.sin(baseDonutSlice.angleSpan / 2) * baseDonutSlice.outRadius)**2)
        newAngleSpan = np.arccos(d / newOutRadius) * 2

        return DonutSlice(xCenter=xNewCenter, yCenter=yNewCenter, inRadius = newInRadius, \
            outRadius=newOutRadius, middleAngle=baseDonutSlice.middleAngle, angleSpan=newAngleSpan)


    def __saveMaps(self, donutSlice, topDistorsionFactor, bottomDistorsionFactor, xMap, yMap):
        np.save(self.__getMapPath(donutSlice, topDistorsionFactor, bottomDistorsionFactor, 'x'), xMap)
        np.save(self.__getMapPath(donutSlice, topDistorsionFactor, bottomDistorsionFactor, 'y'), yMap)


    def __readMaps(self, donutSlice, topDistorsionFactor, bottomDistorsionFactor):
        xMap = np.load(self.__getMapPath(donutSlice, topDistorsionFactor, bottomDistorsionFactor, 'x'))
        yMap = np.load(self.__getMapPath(donutSlice, topDistorsionFactor, bottomDistorsionFactor, 'y'))
        return xMap, yMap


    def __getMapPath(self, donutSlice, topDistorsionFactor, bottomDistorsionFactor, coordinate):
        return os.path.join(rootDirectory, 'config/maps/{xC}-{yC}-{iR}-{oR}-{mA}-{sA}-{tD}-{bD}-{coord}map.npy' \
            .format(xC=int(donutSlice.xCenter), yC=int(donutSlice.yCenter), iR=int(donutSlice.inRadius), \
            oR=int(donutSlice.outRadius), mA=int(np.rad2deg(donutSlice.middleAngle)), \
            sA=int(np.rad2deg(donutSlice.angleSpan)), tD=topDistorsionFactor, bD=bottomDistorsionFactor, coord=coordinate))
    

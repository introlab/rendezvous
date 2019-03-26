from pathlib import Path

import cv2
import numpy as np

from .camera_config import CameraConfig
from src.utils.dewarping_helper import DewarpingHelper
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import FisheyeDewarping
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import DonutSlice


class VideoStream:

    def __init__(self, cameraConfig):
        self.config = CameraConfig(cameraConfig)
        self.dewarper = FisheyeDewarping()
        self.camera = None


    def initializeStream(self):
        self.__initalizeCamera()

        # Changing the codec of the camera throws an exception on camera.read every two execution for whatever reason
        try:
            self.camera.read()
        except:
            self.camera.release()
            self.camera.open(self.config.cameraPort)
            self.__initalizeCamera()

        # Initialization of C++ dewarping library requires number of channels in the video
        success, frame = self.camera.read()

        if not success:
            raise Exception('Could not read image from camera at port {port}'.format(port=self.config.cameraPort))

        channels = len(frame.shape)

        donutSlice = DonutSlice(self.config.imageWidth / 2, self.config.imageHeight / 2, self.config.inRadius, \
            self.config.outRadius, np.deg2rad(self.config.middleAngle), np.deg2rad(self.config.angleSpan))

        dewarpingParameters = DewarpingHelper.getDewarpingParameters(donutSlice, self.config.topDistorsionFactor, self.config.bottomDistorsionFactor)
        self.dewarper.setDewarpingParameters(dewarpingParameters)

        outputWidth = int(dewarpingParameters.dewarpWidth)
        outputHeight = int(dewarpingParameters.dewarpHeight)

        if self.dewarper.initialize(self.config.imageWidth, self.config.imageHeight, outputWidth, outputHeight, channels, True) == -1:
            raise Exception('Error during c++ dewarping library initialization')

        self.printCameraSettings()

        self.dewarpedImageBuffers = np.zeros((2, outputHeight, outputWidth, channels), dtype=np.uint8)
        self.dewarpedImageBuffersIndex = 0

        self.baseDonutSlice = donutSlice
        self.dewarpingParameters = dewarpingParameters
       
    def getDewarpingParameters():
        return self.dewarpingParameters
    
    def getBaseDonutSlice():
        return self.baseDonutSlice

    def destroy(self):
        self.camera.release()
        self.dewarper.cleanUp()


    def readFrame(self):
        success, frame = self.camera.read()

        if success:
            self.dewarper.loadFisheyeImage(frame)
            self.dewarpedImageBuffersIndex = (self.dewarpedImageBuffersIndex + 1) % 2
            self.dewarper.dewarpImage(self.dewarpedImageBuffers[self.dewarpedImageBuffersIndex])
            return success, self.dewarpedImageBuffers[self.dewarpedImageBuffersIndex]
        else:
            return success, None


    def printCameraSettings(self):
        if self.camera == None:
            print('Stream must be initiazed to print the camera settings')
        else:
            print('Image width = {width}'.format(width=self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
            print('Image height = {height}'.format(height=self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print('Codec = {codec}'.format(codec=self.__decode_fourcc(self.camera.get(cv2.CAP_PROP_FOURCC))))
            print('FPS = {fps}'.format(fps=self.camera.get(cv2.CAP_PROP_FPS)))
            print('Buffer size = {bufferSize}'.format(bufferSize=int(self.camera.get(cv2.CAP_PROP_BUFFERSIZE))))


    def __initalizeCamera(self):
        self.camera = cv2.VideoCapture(self.config.cameraPort)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(self.config.fourcc[0],
                                                                    self.config.fourcc[1],
                                                                    self.config.fourcc[2],
                                                                    self.config.fourcc[3]))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.imageWidth)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.imageHeight)
        self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.config.bufferSize)


    # Return 4 chars reprenting codec
    def __decode_fourcc(self, cc):
        return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])
    

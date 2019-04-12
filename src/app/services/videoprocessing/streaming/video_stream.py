from pathlib import Path

import cv2

from .camera_config import CameraConfig
import time


class VideoStream:

    def __init__(self, cameraConfig, useCamera = True):
        self.config = cameraConfig
        self.useCamera = useCamera
        self.camera = None
        self.image = None
        self.imagePath = ''
        self.cameraConnectionRetries = 2


    def initializeStream(self):
        if self.useCamera:
            connectionTries = 0
            success = False

            while connectionTries < self.cameraConnectionRetries and not success:
                try:
                    self.__initalizeCamera()
                    success, frame = self.camera.read()
                except:
                    self.camera.release()
                finally:
                    connectionTries += 1

            if not success:
                raise Exception('Could not read image from camera at port {port}'.format(port=self.config.cameraPort))

            self.printCameraSettings()
        else:
            if self.imagePath == '':
                raise Exception('Path to fisheye image wasn\'t set, make sure to set it when not using a camera')

            self.image = cv2.imread(self.imagePath, cv2.IMREAD_COLOR)
       

    def destroy(self):
        if self.useCamera:
            self.camera.release()


    def readFrame(self):
        if self.useCamera:
            success, frame = self.camera.read()
        else:
            success = True
            frame = self.image.copy()
            time.sleep(0.001)     

        if success:
            return success, frame
        else:
            return success, None


    def printCameraSettings(self):
        if self.camera is None:
            print('Stream must be initiazed to print the camera settings')
        else:
            print('Image width = {width}'.format(width=self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
            print('Image height = {height}'.format(height=self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print('Codec = {codec}'.format(codec=self.__decode_fourcc(self.camera.get(cv2.CAP_PROP_FOURCC))))
            print('FPS = {fps}'.format(fps=self.camera.get(cv2.CAP_PROP_FPS)))
            print('Buffer size = {bufferSize}'.format(bufferSize=int(self.camera.get(cv2.CAP_PROP_BUFFERSIZE))))


    def __initalizeCamera(self):
        self.camera = cv2.VideoCapture(self.config.cameraPort, cv2.CAP_V4L2)
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
    

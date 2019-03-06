from abc import ABC, abstractmethod

class IFaceDetector(ABC):

    # Takes an image as input, returns coordinates of the faces detected in an array
    # return format: [x1, y1, x2, y2]
    @abstractmethod
    def detectFaces(self, image):
        pass


    # Return detection method name
    @abstractmethod
    def getName(self):
        pass
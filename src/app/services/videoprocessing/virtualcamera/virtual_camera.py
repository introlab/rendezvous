import math

from src.utils.rect import Rect
from .face import Face

class VirtualCamera(Rect):

    def __init__(self, face, xPos, yPos, width, height):
        super().__init__(xPos, yPos, width, height)

        self.timeToLive = 5
        self.face = face

        # New position and size to move to every frame (is updated when a new face is associated with this vc) 
        self.positionGoal = xPos, yPos
        self.sizeGoal = width, height


    @staticmethod
    def copy(vc):
        virtualCamera = VirtualCamera(Face.copy(vc.face), vc.xPos, vc.yPos, vc.width, vc.height)
        virtualCamera.timeToLive = vc.timeToLive
        return virtualCamera


    @staticmethod
    def createFromFace(face):
        faceX, faceY = face.getPosition()
        return VirtualCamera(face, faceX, faceY, face.width, face.height)


    def isAlive(self):
        return self.timeToLive > 0


    def decreaseTimeToLive(self):
        self.timeToLive -= 1


    def resetTimeToLive(self):
        self.timeToLive = 5


    


    

import math

from src.utils.spherical_angles_rect import SphericalAnglesRect
from .face import Face

class VirtualCamera(SphericalAnglesRect):

    def __init__(self, face, azimuthLeft, azimuthRight, elevationBottom, elevationTop):
        super().__init__(azimuthLeft, azimuthRight, elevationBottom, elevationTop)

        self.timeToLive = 5
        self.face = face

        # New position and size to move to every frame (is updated when a new face is associated with this vc) 
        self.positionGoal = self.getMiddlePosition()
        self.sizeGoal = self.getAzimuthSpan(), self.getElevationSpan()


    @staticmethod
    def copy(vc):
        virtualCamera = VirtualCamera(Face.copy(vc.face), vc.azimuthLeft, vc.azimuthRight, vc.elevationBottom, vc.elevationTop)
        virtualCamera.timeToLive = vc.timeToLive
        return virtualCamera


    @staticmethod
    def createFromFace(face):
        azimuthLeft, azimuthRight, elevationBottom, elevationTop = face.getAngleCoordinates()
        return VirtualCamera(face, azimuthLeft, azimuthRight, face.elevationBottom, face.elevationTop)


    def isAlive(self):
        return self.timeToLive > 0


    def decreaseTimeToLive(self):
        self.timeToLive -= 1


    def resetTimeToLive(self):
        self.timeToLive = 5
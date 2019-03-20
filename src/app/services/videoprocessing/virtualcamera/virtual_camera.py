import math


class VirtualCamera:

    def __init__(self, xPos, yPos, width, height):
        self.xPos = xPos
        self.yPos = yPos
        self.width = width
        self.height = height
        
        self.timeToLive = 3


    @staticmethod
    def copy(vc):
        virtualCamera = VirtualCamera(vc.xPos, vc.yPos, vc.width, vc.height)
        virtualCamera.timeToLive = vc.timeToLive
        return virtualCamera


    @staticmethod
    def createFromFace(face):
        (facex1, facey1, facex2, facey2) = face
        width = facex2 - facex1
        height = facey2 - facey1
        return VirtualCamera(facex1 + width / 2, facey1 + height / 2, width, height)


    def isAlive(self):
        return self.timeToLive > 0


    def decreaseTimeToLive(self):
        self.timeToLive -= 1


    def resetTimeToLive(self):
        self.timeToLive = 5


    def getBoundingRect(self):
        x1 = math.floor(self.xPos - self.width / 2)
        x2 = math.floor(self.xPos + self.width / 2)
        y1 = math.floor(self.yPos - self.height / 2)
        y2 = math.floor(self.yPos + self.height / 2)
        return (x1, y1, x2, y2)


    def getPosition(self):
        return (self.xPos, self.yPos)


    


    

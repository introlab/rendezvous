import math


class Rect:

    def __init__(self, xPos, yPos, width, height):
        self.xPos = xPos
        self.yPos = yPos
        self.width = width
        self.height = height


    def getBoundingRect(self):
        x1 = math.floor(self.xPos - self.width / 2)
        x2 = math.floor(self.xPos + self.width / 2)
        y1 = math.floor(self.yPos - self.height / 2)
        y2 = math.floor(self.yPos + self.height / 2)
        return (x1, y1, x2, y2)


    def getPosition(self):
        return (self.xPos, self.yPos)
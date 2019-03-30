import numpy as np


class SphericalAnglesRect:

    def __init__(self, azimuthLeft, azimuthRight, elevationBottom, elevationTop):
        self.azimuthLeft = azimuthLeft
        self.azimuthRight = azimuthRight
        self.elevationBottom = elevationBottom
        self.elevationTop = elevationTop


    # Returns the width in angle
    def getAzimuthSpan(self):
        if self.azimuthRight < self.azimuthLeft:
            return ((np.pi * 2) - self.azimuthLeft) + self.azimuthRight
        else:
            return self.azimuthRight - self.azimuthLeft

    
    # Returns the height in angle
    def getElevationSpan(self):
        return self.elevationTop - self.elevationBottom


    # Returns the azimuth and elevation angles of the middle of the rectangle
    def getMiddlePosition(self):
        return ((self.azimuthLeft + self.getAzimuthSpan() / 2) % (np.pi * 2),
                 self.elevationBottom + self.getElevationSpan() / 2)

    
    # Returns the angle coordinate of the 4 sides of the rectangle
    def getAngleCoordinates(self):
        return (self.azimuthLeft,
                self.azimuthRight,
                self.elevationBottom,
                self.elevationTop)

    
    def setAzimuthSpan(self, newAzimuthSpan):
        difference = newAzimuthSpan - self.getAzimuthSpan()
        self.azimuthLeft = (self.azimuthLeft - difference / 2) % (np.pi * 2)
        self.azimuthRight = (self.azimuthRight + difference / 2) % (np.pi * 2)


    def setElevationSpan(self, newElevationSpan):
        difference = newElevationSpan - self.getElevationSpan()
        self.elevationBottom -= difference / 2
        self.elevationTop += difference / 2

    
    def setMiddlePosition(self, newMiddlePosition):
        diffAzimuth = newMiddlePosition[0] - self.getMiddlePosition()[0]
        diffElevation = newMiddlePosition[1] - self.getMiddlePosition()[1]
        self.azimuthLeft = (self.azimuthLeft + diffAzimuth) % (np.pi * 2)
        self.azimuthRight = (self.azimuthRight + diffAzimuth) % (np.pi * 2)
        self.elevationBottom = (self.elevationBottom + diffElevation) % (np.pi * 2)
        self.elevationTop = (self.elevationTop + diffElevation) % (np.pi * 2)

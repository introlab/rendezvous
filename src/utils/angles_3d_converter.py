import math


class Angles3DConverter:

    def __init__(self):
        pass


    @staticmethod
    def azimuthCalculation(x, y):
        return math.atan2(y, x) % (2 * math.pi)


    @staticmethod
    def elevationCalculation(x, y, z):
        xyHypotenuse = math.sqrt(y**2 + x**2)
        return math.atan2(z, xyHypotenuse) % (2 * math.pi)


    @staticmethod
    def degreeToRad(degreeAngle):
        return degreeAngle * math.pi / 180


    @staticmethod
    def radToDegree(radAngle):
        return radAngle * 180 / math.pi

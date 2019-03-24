import math


class GeometryUtils:

    def __init__(self):
        pass


    @staticmethod
    def distanceBetweenTwoPoints(pt1, pt2):
        (x1, y1) = pt1
        (x2, y2) = pt2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    
    @staticmethod
    def getUnitVector(x, y):
        module = math.sqrt(x ** 2 + y ** 2)
        if module == 0:
            return (0, 0)
        return (x / module, y / module)
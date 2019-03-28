from src.utils.rect import Rect

class Face(Rect):
    
    def __init__(self, xPos, yPos, width, height):
        super().__init__(xPos, yPos, width, height)

    @staticmethod
    def copy(face):
        newFace = Face(face.xPos, face.yPos, face.width, face.height)
        return newFace
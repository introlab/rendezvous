from src.utils.spherical_angles_rect import SphericalAnglesRect

class Face(SphericalAnglesRect):
    
    def __init__(self, azimuthLeft, azimuthRight, elevationBottom, elevationTop):
        super().__init__(azimuthLeft, azimuthRight, elevationBottom, elevationTop)

    @staticmethod
    def copy(face):
        newFace = Face(face.azimuthLeft, face.azimuthRight, face.elevationBottom, face.elevationTop)
        return newFace
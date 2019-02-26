import math
import unittest

from src.Geometry.angles_3d_converter import Angles3DConverter

class TestAngles3DConverter(unittest.TestCase):


    def setUp(self):
        pass

        
    def test_azimuthCalculation(self):
        azimuth1 = Angles3DConverter.azimuthCalculation(1, 2)
        azimuth2 = Angles3DConverter.azimuthCalculation(10, 5)
        self.assertAlmostEqual(azimuth1, 1.107, 3)
        self.assertAlmostEqual(azimuth2, 0.464, 3)


    def test_elevationCalculation(self):
        elevation1 = Angles3DConverter.elevationCalculation(1, 2, 10)
        elevation2 = Angles3DConverter.elevationCalculation(5, 10, 11)
        self.assertAlmostEqual(elevation1, 1.35, 2)
        self.assertAlmostEqual(elevation2, 0.78, 2)


    def test_degreeToRad(self):
        angle1 = Angles3DConverter.degreeToRad(180)
        angle2 = Angles3DConverter.degreeToRad(360)
        angle3 = Angles3DConverter.degreeToRad(45)
        angle4 = Angles3DConverter.degreeToRad(100.5)
        self.assertAlmostEqual(angle1, 3.14, 2)
        self.assertAlmostEqual(angle2, 6.28, 2)
        self.assertAlmostEqual(angle3, 0.79, 2)
        self.assertAlmostEqual(angle4, 1.75, 2)


    def test_radToDegree(self):
        angle1 = Angles3DConverter.radToDegree(3.1416)
        angle2 = Angles3DConverter.radToDegree(6.2831)
        angle3 = Angles3DConverter.radToDegree(0.64)
        angle4 = Angles3DConverter.radToDegree(5)
        self.assertAlmostEqual(angle1, 180, 2)
        self.assertAlmostEqual(angle2, 360, 2)
        self.assertAlmostEqual(angle3, 36.67, 2)
        self.assertAlmostEqual(angle4, 286.48, 2)
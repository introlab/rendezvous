import math
import unittest

import numpy as np

from src.utils.spherical_angles_converter import SphericalAnglesConverter


class TestAngles3DConverter(unittest.TestCase):


    def setUp(self):
        pass

        
    def test_azimuthCalculation(self):
       azimuth1 = SphericalAnglesConverter.getAzimuthFromPosition(1, 2)
       azimuth2 = SphericalAnglesConverter.getAzimuthFromPosition(10, 5)
       self.assertAlmostEqual(azimuth1, 1.107, 3)
       self.assertAlmostEqual(azimuth2, 0.464, 3)


    def test_elevationCalculation(self):
       elevation1 = SphericalAnglesConverter.getElevationFromPosition(1, 2, 10)
       elevation2 = SphericalAnglesConverter.getElevationFromPosition(5, 10, 11)
       self.assertAlmostEqual(elevation1, 1.35, 2)
       self.assertAlmostEqual(elevation2, 0.78, 2)


    def test_degreeToRad(self):
        angle1 = np.deg2rad(180)
        angle2 = np.deg2rad(360)
        angle3 = np.deg2rad(45)
        angle4 = np.deg2rad(100.5)
        self.assertAlmostEqual(angle1, 3.14, 2)
        self.assertAlmostEqual(angle2, 6.28, 2)
        self.assertAlmostEqual(angle3, 0.79, 2)
        self.assertAlmostEqual(angle4, 1.75, 2)


    def test_radToDegree(self):
        angle1 = np.rad2deg(3.1416)
        angle2 = np.rad2deg(6.2831)
        angle3 = np.rad2deg(0.64)
        angle4 = np.rad2deg(5)
        self.assertAlmostEqual(angle1, 180, 2)
        self.assertAlmostEqual(angle2, 360, 2)
        self.assertAlmostEqual(angle3, 36.67, 2)
        self.assertAlmostEqual(angle4, 286.48, 2)
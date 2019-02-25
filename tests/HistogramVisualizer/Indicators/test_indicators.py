import unittest
import pathlib
import os
import math

from scripts.HistogramVisualizer.Indicators.indicators import Indicators
from scripts.HistogramVisualizer.FileHelper.file_helper import FileHelper

rootPath = pathlib.Path('../../../').parents[2].absolute()


class TestIndicators(unittest.TestCase):

    def setUp(self):
        eventsPath = os.path.join(rootPath, 'config/testconfigs/ODASOutput.json') 
        if not os.path.exists(eventsPath):
            self.fail('File for testing does not exists at : {path}'.format(path=eventsPath))

        events = FileHelper.readJsonFile(eventsPath)
        self.indicators = Indicators(events)

    
    def test_azimuthCalculation(self):
        azimuth = self.indicators.azimuthCalculation(5, 0)
        self.assertEqual(azimuth * 180 / math.pi, 90)


    def test_elevationCalculation(self):
        elevation = self.indicators.elevationCalculation(5, 0)
        self.assertEqual(elevation * 180 / math.pi, 90)

import unittest
import pathlib
import os

from scripts.ConfigurationHelper.Indicators.indicators import Indicators

rootPath = pathlib.Path('../../../').parents[2].absolute()


class TestIndicators(unittest.TestCase):

    def setUp(self):
        events = []
        srccpath = os.path.join(rootPath, 'config/testconfigs/sources/config1.json')
        if not os.path.exists(srccpath):
            self.fail('config file for testing does not exists at : {path}'.format(path=srccpath))

        self.indicators = Indicators(events, srccpath)

    
    def test_azimuthCalculation(self):
        azimuth = self.indicators.azimuthCalculation(0, 0)
        self.assertEqual(azimuth, 0)

    def test_2(self):
        self.assertFalse('hehe')
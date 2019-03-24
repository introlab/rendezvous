import unittest
from pathlib import Path
import os
import math

from src.tools.configurationhelper.indicators.indicators import Indicators
from src.utils.file_helper import FileHelper

rootPath = str(Path(__file__).resolve().parents[4])


class TestIndicators(unittest.TestCase):

    def setUp(self):
        eventsPath = os.path.join(rootPath, 'config/testconfigs/ODASOutput.json') 
        if not os.path.exists(eventsPath):
            self.fail('File for testing does not exists at : {path}'.format(path=eventsPath))

        srccPath = os.path.join(rootPath, 'config/testconfigs/sources/config1.json')
        if not os.path.exists(srccPath):
            self.fail('File for testing does not exists at : {path}'.format(path=srccPath))
        
        events = FileHelper.readJsonFile(eventsPath)
        config = FileHelper.readJsonFile(srccPath)

        self.indicators = Indicators(events, config)


    # test with empty list
    def test_rms_empty(self):
        try:
            self.indicators.rms(0, [])
        except Exception:
            self.assertRaises(Exception)


    # test with a valid list
    def test_rms_values(self):
        referenceValue = 5
        values = [5, 10, 15, 20, 25, 30, 35, 31, 10]
        rms = self.indicators.rms(referenceValue, values)
        self.assertEqual(rms, 18.184242262647807)

    
    # test core function of Indicators class, just verify there is no crash
    def test_indicatorsCalculation(self):
        try:
            self.indicators.indicatorsCalculation()

        except Exception as e:
            if e:
                self.fail('Exception : {error}'.format(error=e))
                raise e

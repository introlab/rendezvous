import unittest
import pathlib
import os
import collections

from src.tools.histogramvisualizer.histogram.histogram import Histogram

rootPath = pathlib.Path('../../../').parents[2].absolute()


class TestHistogram(unittest.TestCase):
    
    def setUp(self):
        self.histogram = Histogram({'first' : [-1, -2000, -200, 10], 'second': [1, 2, 3]}, 1)


    def test_getMaxValueDict_empty(self):
        dictionary = {'first' : [], 'second': []}
        maxValue = self.histogram.getMaxValueDict(dictionary)        
        self.assertEqual(maxValue, 0)


    def test_getMaxValueDict_one_empty(self):
        dictionary = {'first' : [], 'second': [1, 2, 3]}
        maxValue = self.histogram.getMaxValueDict(dictionary)        
        self.assertEqual(maxValue, 3)

    
    def test_getMaxValueDict_not_empty(self):
        dictionary = {'first' : [-1, -2000, -200, 10], 'second': [1, 2, 3]}
        maxValue = self.histogram.getMaxValueDict(dictionary)        
        self.assertEqual(maxValue, 10)
        

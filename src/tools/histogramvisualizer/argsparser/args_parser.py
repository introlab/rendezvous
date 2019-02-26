import argparse
from os import path

class ArgsParser:

    def __init__(self):
        self.__parser = self.__createParser()
        self.args = self.__parser.parse_args()
        self.validateArgs()


    # arguments parser, all possible arguments are defined here. 
    def __createParser(self):
        parser = argparse.ArgumentParser(description='Histogram and Kernel Density Estimate (KDE) visualization tool based on a file fills with data.')

        parser.add_argument('--datapath', dest='dataPath', action='store', help='path to the data file (json file)', required=True)
        parser.add_argument('--binsrange', dest='binsRange', action='store', type=int, help='range of values for each bins', required=True)
        parser.add_argument('--title', dest='title', action='store', type=str, help='plot title', default='My Very Own Histogram', required=False)
        parser.add_argument('--ylabel', dest='yLabel', action='store', type=str, help='y axis label', default='y', required=False)
        parser.add_argument('--xlabel', dest='xLabel', action='store', type=str, help='x axis label', default='x', required=False)

        return parser


    def validateArgs(self):
        if not path.exists(self.args.dataPath):
            raise self.__parser.error('file at : {path} does not exists.'.format(path=self.args.dataPath))

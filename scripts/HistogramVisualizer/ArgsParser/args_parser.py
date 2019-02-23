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

        return parser


    def validateArgs(self):
        if not path.exists(self.args.dataPath):
            raise self.__parser.error('file at : {path} does not exists.'.format(path=self.args.dataPath))

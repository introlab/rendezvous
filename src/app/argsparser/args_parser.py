import argparse


class ArgsParser:

    def __init__(self):
        self.__parser = self.__createParser()
        self.args = self.__parser.parse_args()
        self.validateArgs()


    # Arguments parser, all possible arguments are defined here. 
    def __createParser(self):
        parser = argparse.ArgumentParser(description='Main app arguments')

        parser.add_argument('--cpath', dest='configPath', action='store', help='Path to the config file for ODAS.')
        parser.add_argument('--opath', dest='odasPath', action='store', help='Path to odaslive program.')
        parser.add_argument('--cameraconfigpath', dest='cameraConfigPath', action='store', help='Path to camera config file.')

        return parser


    def validateArgs(self):
        if not self.args.configPath:
            raise self.__parser.error('--cpath is required.')

        if not self.args.odasPath:
            raise self.__parser.error('--opath is required.')

        if not self.args.cameraConfigPath:
            raise self.__parser.error('--cameraconfigpath is required.')
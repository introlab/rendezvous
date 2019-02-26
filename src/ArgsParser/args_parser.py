import argparse


class ArgsParser:

    def __init__(self):
        self.__parser = self.__createParser()
        self.args = self.__parser.parse_args()
        self.validateArgs()


    # arguments parser, all possible arguments are defined here. 
    def __createParser(self):
        parser = argparse.ArgumentParser(description='Main app arguments')

        parser.add_argument('--cpath', dest='configPath', action='store', help='Path to the config file for ODAS, required if evalconf is not present.')
        parser.add_argument('--opath', dest='odasPath', action='store', help='Path to odaslive program, required if evalconf is not present.')

        return parser


    def validateArgs(self):
        if not self.args.configPath:
            raise self.__parser.error('--cpath is required when --evalconf is set to false')

        if not self.args.odasPath:
            raise self.__parser.error('--opath is required when --evalconf is set to false')

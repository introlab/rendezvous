import argparse


class ArgsParser:

    def __init__(self):
        self.__parser = self.__createParser()
        self.args = self.__parser.parse_args()
        self.validateArgs()


    # arguments parser, all possible arguments are defined here. 
    def __createParser(self):
        parser = argparse.ArgumentParser(description='Helping with 16SoundUSB configurations.')

        parser.add_argument('--evalconf', dest='isEvalConf', action='store_true', help='Activate the validation of configuration.')
        parser.add_argument('--cpath', dest='configPath', action='store', help='Path to the config file for ODAS, required if evalconf is not present.')
        parser.add_argument('--opath', dest='odasPath', action='store', help='Path to odaslive program, required if evalconf is not present.')
        parser.add_argument('--srccpath', dest='sourceConfigPath', action='store', help='Path to the config file for sources positions, required if evalconf is present.')
        parser.add_argument('--cs', dest='chunkSize', action='store', default=500, type=int, help='Backup every chunk of that size usefull if evalconf is present, min value = 500, default value = 500.', required=False)
        parser.add_argument('--time', dest='executionTime', action='store', type=int, default=-1, help='Stream execution time in minutes, please put the time in integers.', required=False)

        return parser


    def validateArgs(self):
        if not self.args.isEvalConf:
            if not self.args.configPath:
                raise self.__parser.error('--cpath is required when --evalconf is set to false')

            if not self.args.odasPath:
                raise self.__parser.error('--opath is required when --evalconf is set to false')

            if self.args.chunkSize < 500:
                raise argparse.ArgumentTypeError('minimal value for --cs is 500')
        else:
            if not self.args.sourceConfigPath:
                raise self.__parser.error('--srccpath is required when --evalconf is set to true')

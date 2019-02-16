import sys
import argparse
import re

from OdasStream.odas_stream import OdasStream
from FileParser.file_parser import FileParser


def createParser():
    parser = argparse.ArgumentParser(description='Helping with 16SoundUSB configurations')

    parser.add_argument('--cfgpath', dest='configPath', action='store', help='Path to the config file for ODAS', required=True)
    parser.add_argument('--odaspath', dest='odasPath', action='store', help='Path to odaslive program', required=True)

    return parser


def main():
    try:
        stream = None
        print('configuration_helper starting...')

        parser = createParser()
        args = parser.parse_args()

        # read config file to get sample rate for while True sleepTime
        line = FileParser.getLineFromFile(args.configPath, 'fS')
        # extract the sample rate from the string and convert to an Integer
        sampleRate = int(re.sub('[^0-9]', '', line.split('=')[1]))
        sleepTime = 1 / sampleRate


        stream = OdasStream(args.odasPath, args.configPath, sleepTime)
        stream.start()

        sys.exit(0)
    
    except Exception as e:
        print('Exception : ', e)
        if stream and stream.subProcess and stream.isRunning :
            stream.stop()

        sys.exit(-1)



if __name__ == '__main__':
    main()

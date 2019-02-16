import sys
import argparse
import re
import signal
import os

from OdasStream.odas_stream import OdasStream
from FileHandler.file_handler import FileHandler

workingDirectory = os.path.dirname(os.path.realpath(sys.argv[0]))

# arguments parser, all possible arguments are defined here. 
def createParser():
    parser = argparse.ArgumentParser(description='Helping with 16SoundUSB configurations')

    parser.add_argument('--cpath', dest='configPath', action='store', help='Path to the config file for ODAS', required=True)
    parser.add_argument('--opath', dest='odasPath', action='store', help='Path to odaslive program', required=True)
    parser.add_argument('--cs', dest='chunkSize', action='store', default=500, type=int, help='backup every chunk of that size, min value = 500, default value = 500', required=False)

    return parser


def validateArgs(args):
    if (args.chunkSize < 500):
        raise argparse.ArgumentTypeError('minimal value for --cs is 500')


# handling for sigterm, sigint, etc.
def signalCallback(event, frameObject):
    exit()


# gracefull exit function.
def exit(exitCode=0):
    print('gracefull exit...')
    if stream and stream.subProcess and stream.isRunning:
        stream.stop()

    sys.exit(exitCode)


def main():
    try:
        print('configuration_helper starting...')
        global stream
        stream = None
        signal.signal(signal.SIGINT, signalCallback)
        signal.signal(signal.SIGTERM, signalCallback)

        # get arguments and validate them.
        parser = createParser()
        args = parser.parse_args()
        validateArgs(args)

        # read config file to get sample rate for while True sleepTime
        line = FileHandler.getLineFromFile(args.configPath, 'fS')
        # extract the sample rate from the string and convert to an Integer
        sampleRate = int(re.sub('[^0-9]', '', line.split('=')[1]))
        sleepTime = 1 / sampleRate

        # start the stream
        options = {'sleepTime' : sleepTime, 'chunkSize' : args.chunkSize}
        stream = OdasStream(args.odasPath, args.configPath, options)
        stream.start()
    
    except Exception as e:
        print('Exception : ', e)

        exitCode = -1
        if stream and stream.subProcess and stream.subProcess.returncode:
            exitCode = stream.subProcess.returncode

        exit(exitCode)


if __name__ == '__main__':
    main()

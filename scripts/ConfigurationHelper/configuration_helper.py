import sys
import argparse
import re
import signal
import os

from OdasStream.odas_stream import OdasStream
from FileHandler.file_handler import FileHandler


workingDirectory = os.path.dirname(os.path.realpath(__file__))


def createParser():
    parser = argparse.ArgumentParser(description='Helping with 16SoundUSB configurations')

    parser.add_argument('--cfgpath', dest='configPath', action='store', help='Path to the config file for ODAS', required=True)
    parser.add_argument('--odaspath', dest='odasPath', action='store', help='Path to odaslive program', required=True)

    return parser


def stopCommand(event, exitCode=0):
    if stream and stream.subProcess and stream.isRunning:
        stream.stop()
    
    if stream:
        fileName = 'ODASOutput.json'
        streamOutputPath = os.path.join(workingDirectory, fileName)
        FileHandler.writeJsonToFile(streamOutputPath, stream.data)

    sys.exit(exitCode)


def main():
    try:
        global stream
        stream = None
        signal.signal(signal.SIGINT, stopCommand)
        signal.signal(signal.SIGTERM, stopCommand)

        print('configuration_helper starting...')

        parser = createParser()
        args = parser.parse_args()

        # read config file to get sample rate for while True sleepTime
        line = FileHandler.getLineFromFile(args.configPath, 'fS')
        # extract the sample rate from the string and convert to an Integer
        sampleRate = int(re.sub('[^0-9]', '', line.split('=')[1]))
        sleepTime = 1 / sampleRate

        stream = OdasStream(args.odasPath, args.configPath, sleepTime)
        stream.start()

    
    except Exception as e:
        print('Exception : ', e)
        stopCommand(-1)


if __name__ == '__main__':
    main()

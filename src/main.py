import sys
import re
import signal
import os

from OdasStream.odas_stream import OdasStream
from FileHelper.file_helper import FileHelper
from ArgsParser.args_parser import ArgsParser


# odas stream instance
global stream
stream = None


# handling for sigterm, sigint, etc.
def signalCallback(event, frameObject):
    exit()


def exit(exitCode=0):
    print('gracefull exit...')
    if stream and stream.odasProcess:
        if stream.isRunning:
            stream.stop()

        if stream.subProcess.returncode:
            exitCode = stream.subProcess.returncode

    sys.exit(exitCode)


def main():
    try:
        # get terminal arguments.
        parser = ArgsParser()
        args = parser.args

        signal.signal(signal.SIGINT, signalCallback)
        signal.signal(signal.SIGTERM, signalCallback)

        # read config file to get sample rate for while True sleepTime
        line = FileHelper.getLineFromFile(args.configPath, 'fS')

        # extract the sample rate from the string and convert to an Integer
        sampleRate = int(re.sub('[^0-9]', '', line.split('=')[1]))
        sleepTime = 1 / sampleRate

        stream = OdasStream(args.odasPath, args.configPath, sleepTime)
        stream.start()

    except Exception as e:
        print('Exception : ', e)
        exit(-1)


if __name__ == '__main__':
    main()

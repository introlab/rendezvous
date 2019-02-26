import sys
import re
import signal
import os

from OdasStream.odas_stream import OdasStream
from OdasStream.odas_stream import AlarmException
from Utils.file_helper import FileHelper
from ArgsParser.args_parser import ArgsParser
from Indicators.indicators import Indicators

workingDirectory = os.path.dirname(os.path.realpath(sys.argv[0]))
outputFilePath = os.path.join(workingDirectory, 'ODASOutput.json')


# odas stream instance
global stream
stream = None


# handling for sigterm, sigint, etc.
def signalCallback(event, frameObject):
    exit()


# gracefull exit function.
def exit(exitCode=0):
    print('gracefull exit...')
    if stream and stream.subProcess:
        if stream.isRunning:
            stream.stop()

        if stream.subProcess.returncode:
            exitCode = stream.subProcess.returncode

    sys.exit(exitCode)


def main():
    try:
        print('configuration_helper starting...')

        # get terminal arguments.
        parser = ArgsParser()
        args = parser.args

        # quality evaluation of the last saved odas data.
        if args.isEvalConf:
            print('indicators evalution starting...')

            # get last saved odas data
            events = FileHelper.readJsonFile(outputFilePath)
            sourceConfigs = FileHelper.readJsonFile(args.sourceConfigPath)

            # calculate indicators
            indicators = Indicators(events, sourceConfigs)
            indicators.indicatorsCalculation()


        # ODAS stream
        else:
            # delete output file
            if os.path.exists(outputFilePath):
                os.remove(outputFilePath)

            signal.signal(signal.SIGINT, signalCallback)
            signal.signal(signal.SIGTERM, signalCallback)
            # read config file to get sample rate for while True sleepTime
            line = FileHelper.getLineFromFile(args.configPath, 'fS')
            # extract the sample rate from the string and convert to an Integer
            sampleRate = int(re.sub('[^0-9]', '', line.split('=')[1]))
            sleepTime = 1 / sampleRate

            # start the stream
            options = {'chunkSize' : args.chunkSize}
            stream = OdasStream(args.odasPath, args.configPath, sleepTime, options)
            stream.start(args.executionTime)
    
    except AlarmException as e:
        print('Execution Timeout')
        exit(0)

    except Exception as e:
        print('Exception : ', e)
        exit(-1)


if __name__ == '__main__':
    main()

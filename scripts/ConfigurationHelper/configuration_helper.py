import sys
import argparse
import re

from OdasStream.odas_stream import OdasStream


def createParser():
    parser = argparse.ArgumentParser(description='Helping with 16SoundUSB configurations')

    parser.add_argument('--cfgPath', dest='configPath', action='store', help='Path to the config file for ODAS', required=True)
    parser.add_argument('--odasPath', dest='odasPath', action='store', help='Path to odaslive program', required=True)

    return parser

def getLineFromFile(filePath, startsWith):
    with open(filePath, 'r') as fi:
        for line in fi:
            stripedLine = line.replace(' ', '')
            if stripedLine.startswith(startsWith):
                return stripedLine


def main():
    try:
        stream = None
        print('configuration_helper starting...')

        parser = createParser()
        args = parser.parse_args()

        # read the config file to get the sample rate
        line = getLineFromFile(args.configPath, 'fS')
        sampleTime = int(re.sub('[^0-9]', '', line.split('=')[1]))
        sleepTime = 1 / sampleTime


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

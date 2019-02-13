import sys
import argparse

from OdasStream.odas_stream import OdasStream


def createParser():
    parser = argparse.ArgumentParser(description='Helping with 16SoundUSB configurations')

    parser.add_argument('--cfgPath', dest='configPath', action='store', help='Path to the config file for ODAS', required=True)
    parser.add_argument('--odasPath', dest='odasPath', action='store', help='Path to odaslive program', required=True)

    return parser


def main():
    try:
        print('configuration_helper starting...')

        parser = createParser()
        args = parser.parse_args()

        stream = OdasStream(args.odasPath, args.configPath)
        stream.start()
        
        sys.exit(0)
    
    except Exception as e:
        print('Exception : ', e)
        if stream.subProcess:
            stream.stop()

        sys.exit(-1)



if __name__ == '__main__':
    main()

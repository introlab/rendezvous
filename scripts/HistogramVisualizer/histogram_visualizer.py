import sys
import os

from FileHelper.file_helper import FileHelper
from ArgsParser.args_parser import ArgsParser

workingDirectory = os.path.dirname(os.path.realpath(sys.argv[0]))

def main():
    try:
        print('histogram_visualizer starting...')

        # get terminal arguments.
        parser = ArgsParser()
        args = parser.args

        print('maman ca part')

    except Exception as e:
        print('Exception : ', e)
        sys.exit(-1)


if __name__ == '__main__':
    main()

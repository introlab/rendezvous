import sys
import os

from FileHelper.file_helper import FileHelper
from ArgsParser.args_parser import ArgsParser
from Indicators.indicators import Indicators
from Histogram.histogram import Histogram

workingDirectory = os.path.dirname(os.path.realpath(sys.argv[0]))

def main():
    try:
        print('histogram_visualizer starting...')

        # get terminal arguments.
        parser = ArgsParser()
        args = parser.args

        # load data
        events = FileHelper.readJsonFile(args.dataPath)
        indicators = Indicators(events)
        data = {'azimuth' : indicators.azimuths, 'elevation' : indicators.elevations}

        hist = Histogram(data['azimuth'])
        hist.plot(title=args.title, xLabel=args.xLabel, yLabel=args.yLabel)

    except Exception as e:
        print('Exception : ', e)
        raise e


if __name__ == '__main__':
    main()

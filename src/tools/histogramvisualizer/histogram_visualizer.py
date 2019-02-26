import sys
import os
import collections

from utils.file_helper import FileHelper
from argsparser.args_parser import ArgsParser
from indicators.indicators import Indicators
from histogram.histogram import Histogram


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
        data = collections.OrderedDict({'azimuth' : indicators.azimuths, 'elevation' : indicators.elevations})

        # create an histogram and show a plot
        hist = Histogram(data, args.binsRange)
        hist.plotWithDensity(title=args.title, xLabel=args.xLabel, yLabel=args.yLabel)

    except Exception as e:
        print('Exception : ', e)
        raise e


if __name__ == '__main__':
    main()

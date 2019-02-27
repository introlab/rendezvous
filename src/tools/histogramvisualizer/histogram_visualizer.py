import sys
import os
import collections

workingDirectory = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.join(workingDirectory, '../../../src'))

import context
from src.utils.file_helper import FileHelper
from src.tools.histogramvisualizer.argsparser.args_parser import ArgsParser
from src.tools.histogramvisualizer.indicators.indicators import Indicators
from src.tools.histogramvisualizer.histogram.histogram import Histogram


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

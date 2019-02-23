import matplotlib.pyplot as plt
import pandas
import numpy

class Histogram:

    def __init__(self, data, binsRange):
        self.histogramData = data
        self.__bins = []
        self.__calculateBins(binsRange)


    @staticmethod
    def getMaxValueDict(dictionary):
        maxValue = 0
        for _, values in dictionary.items():
            itemMaxValue = max(values)
            if (maxValue < itemMaxValue):
                maxValue = itemMaxValue

        return maxValue


    def plotWithDensity(self, xLabel='x', yLabel='y', title='My Very Own Histogram'):
        columns = []
        for key, _ in self.histogramData.items():
            columns.append(key)

        dist = pandas.DataFrame(self.histogramData, columns=columns)
        _, ax = plt.subplots()
        dist.plot.hist(bins=self.__bins, density=True, ax=ax)
        dist.plot.kde(ax=ax, legend=False, title=title)

        ax.set_ylabel(yLabel)
        ax.set_xlabel(xLabel)
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.grid(axis='y', alpha=0.75)
        ax.set_facecolor('#d8dcd6')
        plt.show()


    def __calculateBins(self, binsRange):
        self.bins = []
        maxValue = self.getMaxValueDict(self.histogramData)
        stopValue = int(maxValue) + 1
        step = int(binsRange)
        for value in range (0, stopValue, step):
            self.__bins.append(value)

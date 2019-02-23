import matplotlib.pyplot as plt
import numpy

class Histogram:

    def __init__(self, data):
        self.histogramData = data
        n, bins, patches = plt.hist(x=self.histogramData, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85, density=True)
        self.n = n
        self.bins = bins
        self.patches = patches


    def plot(self, xLabel='x', yLabel='y', title='My Very Own Histogram'):
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        maxfreq = self.n.max()
        plt.show()


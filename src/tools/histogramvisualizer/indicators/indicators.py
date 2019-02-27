import math

from src.utils.angles_3d_converter import Angles3DConverter


class Indicators:

    def __init__(self, events):
        self.events = events
        self.azimuths = []
        self.elevations = []
        self.__calculateIndicators()


    def __calculateIndicators(self):
        # calculate indicators for each events
        for event in self.events:
            sources = event[0]['src']
            for _, source in sources.items():

                x = source['x']
                y = source['y']
                z = source['z']
                azimuth = Angles3DConverter.azimuthCalculation(x, y)
                elevation = Angles3DConverter.elevationCalculation(x, y, z)
                        
                self.azimuths.append(Angles3DConverter.radToDegree(azimuth))
                self.elevations.append(Angles3DConverter.radToDegree(elevation))
                
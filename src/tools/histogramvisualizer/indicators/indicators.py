import math

import numpy as np

from src.utils.spherical_angles_converter import SphericalAnglesConverter


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
                azimuth = SphericalAnglesConverter.getAzimuthFromPosition(x, y)
                elevation = SphericalAnglesConverter.getElevationFromPosition(x, y, z)
                        
                self.azimuths.append(np.rad2deg(azimuth))
                self.elevations.append(np.rad2deg(elevation))
                
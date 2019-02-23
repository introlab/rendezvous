import math

class Indicators:

    def __init__(self, events):
        self.events = events
        self.azimuths = []
        self.elevations = []
        self.__calculateIndicators()

    
    def azimuthCalculation(self, y, x):
        azimuth = math.atan2(y, x) % (2 * math.pi)
        return azimuth


    def elevationCalculation(self, z, xyHypotenuse):
        elevation = math.atan2(z, xyHypotenuse) % (2 * math.pi)
        return elevation


    def __calculateIndicators(self):
        # calculate indicators for each events
        for event in self.events:
            sources = event[0]['src']
            for _, source in sources.items():

                x = source['x']
                y = source['y']
                z = source['z']
                xyHypotenuse = math.sqrt(y**2 + x**2)
                azimuth = self.azimuthCalculation(y, x) * 180 / math.pi
                elevation = self.elevationCalculation(z, xyHypotenuse) * 180 / math.pi
                        
                self.azimuths.append(azimuth)
                self.elevations.append(elevation)
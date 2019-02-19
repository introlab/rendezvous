import math

class Indicators:

    def __init__(self, events):
        self.events = events
        self.azimuth = {'sum' : [0, 0, 0, 0], 'average' : [0, 0, 0, 0]}
        self.elevation = {'sum' : [0, 0, 0, 0], 'average' : [0, 0, 0, 0]}


    def indicatorsCalculation(self):
        self.__sumCalculation()
        self.__averageCalculation()


    def __sumCalculation(self):
        # calculate indicators for each events
            for event in self.events:
                sources = event[0]['src']

                for index, source in enumerate(sources):
                    x = source['x']
                    y = source['y']
                    z = source['z']
                    xyHypotenuse = math.sqrt(y**2 + x**2)
                    azimuth = math.atan2(y, x)
                    elevation = math.atan2(z, xyHypotenuse)
                    self.azimuth['sum'][index] += azimuth
                    self.elevation['sum'][index] += elevation


    def __averageCalculation(self):
        eventsLength = len(self.events)
        # average azimuth for each source
        print('\n\nAzimuth for each source (degree) :\n')
        for index, azimuthSum in enumerate(self.azimuth['sum']):
            avgAzimuth = azimuthSum / eventsLength * 360 / 2*math.pi
            self.azimuth['average'][index] = avgAzimuth
            print('source {sourceNumber} : {azimuth}'.format(sourceNumber=(index + 1), azimuth=avgAzimuth))

        # average elevation for each source
        print('\n\nElevation for each source (degree) :\n')
        for index, elevationSum in enumerate(self.elevation['sum']):
            avgElevation = elevationSum / eventsLength * 360 / 2*math.pi
            self.elevation['average'][index] = avgElevation 
            print('source {sourceNumber} : {elevation}'.format(sourceNumber=(index + 1), elevation=avgElevation))

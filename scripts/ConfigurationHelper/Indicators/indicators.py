import math

class Indicators:

    def __init__(self, events, config):
        self.events = events
        self.config = config
        self.eventsPerSrc = [0, 0, 0, 0]
        self.azimuth = {'sum' : [0, 0, 0, 0], 'average' : [0, 0, 0, 0]}
        self.elevation = {'sum' : [0, 0, 0, 0], 'average' : [0, 0, 0, 0]}
        self.configSources = []


    def indicatorsCalculation(self):
        self.__initializeConfigSources()
        self.__sumCalculation()
        self.__averageCalculation()


    def __initializeConfigSources(self):
        detectionArea = self.config['DetectionArea']
        width = detectionArea['Width']
        height = detectionArea['Height']

        for source in self.config['Sources']:
            x = source['x']
            y = source['y']
            z = source['z']

            # Calculate the acceptable range of the azimuth angle
            xyHypotenuse = math.sqrt(x**2 + y**2)
            dThetaAz = math.sin(width / (2 * xyHypotenuse))
            azimuth = math.atan2(y, x) % (2 * math.pi)
            azimuthBounds = { 'min' : azimuth - dThetaAz, 'max' : azimuth + dThetaAz }

            # Calculate the acceptable range of the elevation angle
            xyzHypotenuse = math.sqrt(x**2 + y**2 + z**2)
            dThetaEl = math.sin(height / (2* xyzHypotenuse))
            elevation = math.atan2(z, xyHypotenuse) % (2 * math.pi)
            elevationBounds = { 'min' : elevation - dThetaEl, 'max' : elevation + dThetaEl }

            self.configSources.append({'x' : x, 'y' : y, 'z' : z, 
                'azimuthBounds' : azimuthBounds, 'elevationBounds' : elevationBounds})


    def __sumCalculation(self):
        # calculate indicators for each events
        for event in self.events:
            sources = event[0]['src']
            for key, source in sources.items():
                sourceId = int(key)
                self.eventsPerSrc[sourceId - 1] += 1
                x = source['x']
                y = source['y']
                z = source['z']
                xyHypotenuse = math.sqrt(y**2 + x**2)
                azimuth = (math.atan2(y, x) * 180 / math.pi) % 360 
                elevation = (math.atan2(z, xyHypotenuse) * 180 / math.pi) % 360
                self.azimuth['sum'][sourceId - 1] += azimuth
                self.elevation['sum'][sourceId - 1] += elevation


    def __averageCalculation(self):
        # average azimuth for each source
        print('\n\nAzimuth for each source (degree) :\n')
        for index, azimuthSum in enumerate(self.azimuth['sum']):
            if self.eventsPerSrc[index] != 0:
                avgAzimuth = azimuthSum / self.eventsPerSrc[index]
                self.azimuth['average'][index] = avgAzimuth
                print('source {sourceNumber} : {azimuth}'.format(sourceNumber=(index + 1), azimuth=avgAzimuth))

        # average elevation for each source
        print('\n\nElevation for each source (degree) :\n')
        for index, elevationSum in enumerate(self.elevation['sum']):
            if self.eventsPerSrc[index] != 0:
                avgElevation = elevationSum / self.eventsPerSrc[index]
                self.elevation['average'][index] = avgElevation 
                print('source {sourceNumber} : {elevation}'.format(sourceNumber=(index + 1), elevation=avgElevation))

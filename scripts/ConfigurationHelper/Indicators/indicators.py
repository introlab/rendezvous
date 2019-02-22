import math

class Indicators:

    def __init__(self, events, config):
        self.events = events
        self.config = config
        self.eventsPerSrc = [0, 0, 0, 0]
        self.positionTest = {'inBounds' : [0, 0, 0, 0], 'total' : [0, 0, 0, 0]}
        self.azimuth = {
            'values' : {'0' : [], '1' : [], '2' : [], '3' : []},
            'sum' : [0, 0, 0, 0],
            'average' : [0, 0, 0, 0],
            'rms' : [0, 0, 0, 0]
        }
        self.elevation = {
            'values' : {'0' : [], '1' : [], '2' : [], '3' : []},
            'sum' : [0, 0, 0, 0],
            'average' : [0, 0, 0, 0],
            'rms' : [0, 0, 0, 0]
        }
        self.configSources = []


    def indicatorsCalculation(self):
        self.__initializeConfigSources()
        self.__processEvents()
        self.__averageCalculation()
        self.__rateDetectionInBounds()
        self.__falseDetectionRate()
        self.__rmsCalculation()

        
    def __initializeConfigSources(self):
        if not 'Sources' in self.config or len(self.config['Sources']) == 0:
            raise Exception('Missing sources in config')

        if not 'DetectionArea' in self.config:
            raise Exception('Missing detection area in config')

        detectionArea = self.config['DetectionArea']
        width = detectionArea['Width']
        height = detectionArea['Height']

        print('\nReel angles for each sources (degree) : \n')
        for index, source in enumerate(self.config['Sources']):
            x = source['x']
            y = source['y']
            z = source['z']

            # Calculate the acceptable range of the azimuth angle
            xyHypotenuse = math.sqrt(x**2 + y**2)
            dThetaAz = math.sin(width / (2 * xyHypotenuse))
            azimuth = self.__azimuthCalculation(y, x)
            print('Reel azimuth for source {source} equals = {azimuth}'.format(source=index+1, azimuth=azimuth * 180 / math.pi))
            azimuthBounds = { 'min' : azimuth - dThetaAz, 'max' : azimuth + dThetaAz }

            # Calculate the acceptable range of the elevation angle
            xyzHypotenuse = math.sqrt(x**2 + y**2 + z**2)
            dThetaEl = math.sin(height / (2* xyzHypotenuse))
            elevation = self.__elevationCalculation(z, xyHypotenuse)
            print('Reel elevation for source {source} equals = {elevation}'.format(source=index+1, elevation=elevation * 180 / math.pi))
            elevationBounds = { 'min' : elevation - dThetaEl, 'max' : elevation + dThetaEl }

            self.configSources.append({'x' : x, 'y' : y, 'z' : z, 
                'azimuthBounds' : azimuthBounds, 'elevationBounds' : elevationBounds})


    def __processEvents(self):
        # calculate indicators for each events
        for event in self.events:
            sources = event[0]['src']
            for _, source in sources.items():

                index = self.__getConfigSourceIndex(source)
                if index == None:
                    raise Exception("can't associate this ODAS source with a real source")

                indexStr = str(index)

                self.eventsPerSrc[index] += 1
                x = source['x']
                y = source['y']
                z = source['z']
                xyHypotenuse = math.sqrt(y**2 + x**2)
                azimuth = self.__azimuthCalculation(y, x)
                elevation = self.__elevationCalculation(z, xyHypotenuse)

                azimuthBounds = self.configSources[index]['azimuthBounds']
                elevationBounds = self.configSources[index]['elevationBounds']

                # Check if source position is inside the configured source bounds
                if azimuth >= azimuthBounds['min'] and azimuth <= azimuthBounds['max'] and \
                    elevation >= elevationBounds['min'] and elevation <= elevationBounds['max'] :
                    self.positionTest['inBounds'][index] += 1
                        
                self.positionTest['total'][index] += 1
                self.azimuth['sum'][index] += azimuth
                self.azimuth['values'][indexStr].append(azimuth)
                self.elevation['sum'][index] += elevation
                self.elevation['values'][indexStr].append(elevation)


    def __getConfigSourceIndex(self, source):
        minDistance = -1
        index = None

        for configSource in self.configSources:
            dx = configSource['x'] - source['x']
            dy = configSource['y'] - source['y']
            dz = configSource['z'] - source['z']
            distance = math.sqrt(dx**2 + dy**2 + dz**2)

            if distance < minDistance or minDistance == -1:
                minDistance = distance
                index = self.configSources.index(configSource)

        return index


    def __averageCalculation(self):
        # average azimuth for each source
        print('\n\nAzimuth for each source (degree) :\n')
        for index, azimuthSum in enumerate(self.azimuth['sum']):
            if self.eventsPerSrc[index] != 0:
                avgAzimuth = azimuthSum / self.eventsPerSrc[index]
                self.azimuth['average'][index] = avgAzimuth
                avgAzimuthDegree = (avgAzimuth * 180 / math.pi) % 360
                print('source {sourceNumber} : {azimuth}'.format(sourceNumber=(index + 1), azimuth=(avgAzimuthDegree)))

        # average elevation for each source
        print('\n\nElevation for each source (degree) :\n')
        for index, elevationSum in enumerate(self.elevation['sum']):
            if self.eventsPerSrc[index] != 0:
                avgElevation = elevationSum / self.eventsPerSrc[index]
                self.elevation['average'][index] = avgElevation
                avgElevationDegree = (avgElevation * 180 / math.pi) % 360
                print('source {sourceNumber} : {elevation}'.format(sourceNumber=(index + 1), elevation=avgElevationDegree))


    def __rateDetectionInBounds(self):
        print('\n\nIn bounds detection rate (%) :\n')
        for index, total in enumerate(self.positionTest['total']):
            if self.eventsPerSrc[index] != 0:
                detectionRate = self.positionTest['inBounds'][index] / total
                print('source {sourceNumber} : {rate}'.format(sourceNumber=(index + 1), rate=detectionRate))


    def __falseDetectionRate(self):
        print('\n\nRate of failed detections:\n')
        firstEventNb = self.events[0][0]['timestamp']
        lastEventNb = self.events[-1][0]['timestamp']
        totalEvents = lastEventNb - firstEventNb + 1
        for index, nbEvents in  enumerate(self.eventsPerSrc):
            if index < len(self.configSources) :
                print('source {sourceNumber} : {rate}'.format(sourceNumber=(index + 1), rate=(totalEvents - nbEvents) / totalEvents))


    def __rmsCalculation(self):
        print('\n\nAzimuth RMS for each source :\n')
        for key, azimuthValues in self.azimuth['values'].items():
            if azimuthValues != [] and int(key) < len(self.config['Sources']):
                x = self.config['Sources'][int(key)]['x']
                y = self.config['Sources'][int(key)]['y']
                azimuthOfReference = self.__azimuthCalculation(y, x)
                rms = self.__rms(azimuthOfReference, azimuthValues)
                self.azimuth['rms'][int(key)] = rms
                print('source {sourceNumber} : {rms}'.format(sourceNumber=(int(key) + 1), rms=rms))

        print('\n\nElevation RMS for each source :\n')
        for key, elevationValues in self.elevation['values'].items():
            if elevationValues != [] and int(key) < len(self.config['Sources']):
                x = self.config['Sources'][int(key)]['x']
                y = self.config['Sources'][int(key)]['y']
                z = self.config['Sources'][int(key)]['z']
                xyHypotenuse = math.sqrt(x**2 + y**2 + z**2)
                elevationOfReference = self.__azimuthCalculation(z, xyHypotenuse)
                rms = self.__rms(elevationOfReference, elevationValues)
                self.elevation['rms'][int(key)] = rms
                print('source {sourceNumber} : {rms}'.format(sourceNumber=(int(key) + 1), rms=rms))


    def __azimuthCalculation(self, y, x):
        azimuth = math.atan2(y, x) % (2 * math.pi)
        return azimuth


    def __elevationCalculation(self, z, xyHypotenuse):
        elevation = math.atan2(z, xyHypotenuse) % (2 * math.pi)
        return elevation


    def __rms(self, valueOfReference, values=[]):
        sumOfSquare = 0
        rms = 0

        if (values == []):
            raise Exception("can't calculate rms without values ;)")

        for value in values:
            sumOfSquare += (valueOfReference - value)**2

        n = len(values)
        rms = math.sqrt(sumOfSquare / n)

        return rms

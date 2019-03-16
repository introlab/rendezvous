

class CameraConfig:

    def __init__(self, cameraConfigJson):
        self.__validateConfig(cameraConfigJson)

        self.imageWidth = cameraConfigJson['Image']['Width']
        self.imageHeight = cameraConfigJson['Image']['Height']
        self.fisheyeAngle = cameraConfigJson['Image']['FisheyeAngle']
            
        self.cameraPort = cameraConfigJson['Camera']['Port']
        self.fps = cameraConfigJson['Camera']['FPS']
        self.fourcc = cameraConfigJson['Camera']['Fourcc']
        self.bufferSize = cameraConfigJson['Camera']['BufferSize']
        
        self.topDistorsionFactor = cameraConfigJson['Distorsion']['TopDistorsionFactor']
        self.bottomDistorsionFactor = cameraConfigJson['Distorsion']['BottomDistorsionFactor']
        self.inRadius = cameraConfigJson['Distorsion']['InRadius']
        self.outRadius = cameraConfigJson['Distorsion']['OutRadius']
        self.middleAngle = cameraConfigJson['Distorsion']['MiddleAngle']
        self.angleSpan = cameraConfigJson['Distorsion']['AngleSpan']


    def __validateConfig(self, cameraConfig):

        if 'Image' in cameraConfig:
            imageParams = ['Width', 'Height']
            if imageParams.sort() != list(cameraConfig['Image'].keys()).sort():
                raise Exception('Make sure the following Image parameters are set in the config: ', imageParams)
        else:
            raise Exception('Missing Image params in config')

        if 'Camera' in cameraConfig:
            cameraParams = ['Port', 'FPS', 'Fourcc']
            if cameraParams.sort() != list(cameraConfig['Camera'].keys()).sort():
                raise Exception('Make sure the following Camera parameters are set in the config: ', cameraParams)
        else:
            raise Exception('Missing Camera params in config')

        if 'Distorsion' in cameraConfig:
            distorsionParams = ['TopDistorsionFactor', 'BottomDistorsionFactor', 'InRadius', 'OutRadius', 'MiddleAngle', 'AngleSpan']
            if distorsionParams.sort() != list(cameraConfig['Distorsion'].keys()).sort():
                raise Exception('Make sure the following Distorsion parameters are set in the config: ', distorsionParams)
        else:
            raise Exception('Missing Distorsion params in config')



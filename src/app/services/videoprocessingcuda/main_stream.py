import numpy as np
import time

import context
from src.app.services.videoprocessingcuda.interface.stream import Stream
from src.app.services.videoprocessingcuda.interface.stream import CameraConfig
from src.app.services.videoprocessingcuda.interface.stream import DewarpingConfig


def main():
    inRadius = 400
    outRadius = 1400
    angleSpan = np.deg2rad(90)
    topDistorsionFactor = 0.08
    bottomDistorsionFactor = 0
    fisheyeAngle = np.deg2rad(220)
    minElevation = np.deg2rad(0)
    maxElevation = np.deg2rad(90)
    aspectRatio = 3 / 4
    dewarpingConfig = DewarpingConfig(inRadius, outRadius, angleSpan,topDistorsionFactor, bottomDistorsionFactor, fisheyeAngle)

    width = 2880
    height = 2160
    channels = 3
    fpsTarget = 20
    cameraConfig = CameraConfig(width, height, channels, fpsTarget)

    stream = Stream(cameraConfig, dewarpingConfig)
    stream.start()
    time.sleep(1)
    stream.stop()


if __name__ == '__main__':
	main()
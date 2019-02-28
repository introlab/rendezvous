import numpy as np

from streaming.video_stream import VideoStream
from streaming.video_stream import DonutSlice


def main():

    # Image size
    width = 2880
    heigth = 2160

    # Factor near zero means the dewarped image will follow the fisheye image curvature
    topDistorsionFactor = 0.08
    bottomDistorsionFactor = 0

    # Circles where the image cropping occurs (center of image to inRadius and outRadius to image borders will be cropped)
    inRadius = 400
    outRadius = 1400

    # 0 degree is bottom of source image, sets angular region to be dewarped
    middleAngle = 90 
    angleSpan = 90

    donutSlice = DonutSlice(xCenter=width / 2, yCenter=heigth / 2, inRadius = inRadius, \
        outRadius=outRadius, middleAngle=np.deg2rad(middleAngle), angleSpan=np.deg2rad(angleSpan))

    stream = VideoStream(1, width, heigth, 20, 'UYVY')
    
    # If no maps are found for your parameters, they will be generated
    stream.startStream(donutSlice, topDistorsionFactor, bottomDistorsionFactor, False)


if __name__ == '__main__':
    main()
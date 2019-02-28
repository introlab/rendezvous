import numpy as np

from streaming.video_stream import VideoStream
from streaming.video_stream import DonutMapping


def main():

    # Image size
    width = 2880
    heigth = 2160

    # Will modify the level of distorsion mostly in the top region of the image
    centersDistance = 800

    # Circles where the image cropping occurs (center of image to inRadius and outRadius to image borders will be cropped)
    inRadius = 400
    outRadius = 1400

    # 0 degree is bottom of source image, sets angular region to be dewarped
    middleAngle = 90 
    angleSpan = 90

    donutMapping = DonutMapping(xCenter=width / 2, yCenter=heigth / 2, inRadius = inRadius, \
        outRadius=outRadius, middleAngle=np.deg2rad(middleAngle), angleSpan=np.deg2rad(angleSpan))

    stream = VideoStream(1, width, heigth, 20, 'UYVY')
    
    # If no maps are found for your parameters, they will be generated
    stream.startStream(donutMapping, centersDistance, False) 


if __name__ == '__main__':
    main()
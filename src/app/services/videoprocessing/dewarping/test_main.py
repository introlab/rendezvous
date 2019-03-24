import time
from collections import namedtuple

import cv2
import numpy as np

import context
from src.utils.dewarping_helper import DewarpingHelper
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import FisheyeDewarping
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import DonutSlice
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import DewarpingParameters

DebugImageInfoParam = namedtuple('DebugImageInfoParam', 'donutSlice \
    newDonutSlice center newCenter bottomLeft bottomRight centerRadius')


def main():

	debug = False

	# Camera desired parameters
	port = 1
	requestWidth = 2880
	requestHeight = 2160

	# Try to modify camera parameters
	cam = cv2.VideoCapture(port)
	cam.set(cv2.CAP_PROP_FRAME_WIDTH, requestWidth)
	cam.set(cv2.CAP_PROP_FRAME_HEIGHT, requestHeight)

	# Camera image actual size
	width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
	channels = len(cam.read()[1].shape)

	# Output size
	outputWidth = int(1280 / 2)
	outputHeight = int(720 / 2)

	# Factor near zero means the dewarped image will follow the fisheye image curvature
	topDistorsionFactor = 0.08
	bottomDistorsionFactor = 0

	# Circles where the image cropping occurs (center of image to inRadius and outRadius to image borders will be cropped)
	inRadius = 400
	outRadius = 1400

	# 0 degree is bottom of source image, sets angular region to be dewarped
	middleAngle = 90
	angleSpan = 90

	# Variables to get same dewarping but with less data from source image on y axis (to create virtual cameras)
	topOffset = 200
	bottomOffset = 200

	outputHeightVC = 200
	outputWidthVC = 100

	donutSlice = DonutSlice(width / 2.0, height / 2.0, inRadius, outRadius, np.deg2rad(middleAngle), np.deg2rad(angleSpan))
	donutSlice2 = DonutSlice(width / 2.0, height / 2.0, inRadius, outRadius, np.deg2rad(middleAngle), np.deg2rad(angleSpan - 45))
	donutSlice3 = DonutSlice(width / 2.0, height / 2.0, inRadius + topOffset, outRadius, np.deg2rad(middleAngle), np.deg2rad(angleSpan - 45))

	outputPortraitWidth = int(outputWidth / 2)
	outputPortraitHeight = int(outputHeight * ((outRadius - inRadius - topOffset - bottomOffset) / (outRadius - inRadius)))

	dewarpedImage = np.empty((outputHeight, outputWidth, channels), dtype=np.uint8)
	dewarpedPortrait = np.empty((outputPortraitHeight, outputPortraitWidth, channels), dtype=np.uint8)

	dewarper = FisheyeDewarping(width, height, channels, True)
	bufferId = dewarper.createRenderContext(outputWidth, outputHeight, channels)
	bufferPortraitId = dewarper.createRenderContext(outputPortraitWidth, outputPortraitHeight, channels)

	dewarpedImages = {}
	dewarpedImages[bufferId] = dewarpedImage
	dewarpedImages[bufferPortraitId] = dewarpedPortrait

	dewarpingParameters = DewarpingHelper.getDewarpingParameters(donutSlice, topDistorsionFactor, bottomDistorsionFactor)
	dewarpingParameters.topOffset = 0
	dewarpingParameters.bottomOffset = 0
	dewarpingParameters2 = DewarpingHelper.getDewarpingParameters(donutSlice2, topDistorsionFactor, bottomDistorsionFactor)
	dewarpingParameters2.topOffset = topOffset
	dewarpingParameters2.bottomOffset = bottomOffset

	rect = (dewarpingParameters.dewarpWidth / 4, 0, (dewarpingParameters.dewarpWidth * 3) / 4, dewarpingParameters.dewarpHeight)
	vcDewarpingParameters = DewarpingHelper.getVirtualCameraDewarpingParameters(rect, donutSlice, dewarpingParameters, topDistorsionFactor)

	debugImageInfoParam = None

	if debug:
		debugImageInfoParam = createDebugImageInfoParam(donutSlice, topDistorsionFactor)
		debugImageInfoParam2 = createDebugImageInfoParam(donutSlice3, topDistorsionFactor)

	while True:
		success, frame = cam.read()

		if success:

			if debugImageInfoParam:
				addDebugInfoToImage(frame, debugImageInfoParam)
				addDebugInfoToImage(frame, debugImageInfoParam2)

			dewarper.loadFisheyeImage(frame)

			dewarper.queueDewarping(bufferId, dewarpingParameters, dewarpedImage)
			dewarper.queueDewarping(bufferPortraitId, dewarpingParameters2, dewarpedPortrait)

			buffer = 0
			while buffer != -1:
				cv2.imshow("{buffer}".format(buffer=buffer), dewarpedImages[buffer])
				buffer = dewarper.dewarpNextImage()

			cv2.imshow('360', cv2.resize(frame, (640, 480)))

		k = cv2.waitKey(1)
		if k & 0xFF == ord('w'):
			topDistorsionFactor += 0.005
		if k & 0xFF == ord('s'):
			topDistorsionFactor -= 0.005
		if k & 0xFF == ord('d'):
			donutSlice.middleAngle += 0.02
		if k & 0xFF == ord('a'):
			donutSlice.middleAngle -= 0.02
		if k & 0xFF == 27:
			break


# Create the dataset required to display the debug lines on the source image
def createDebugImageInfoParam(donutSlice, topDistorsionFactor):
	centersDistance = topDistorsionFactor * 10000
	newDonutSlice = DewarpingHelper.createDewarpingDonutSlice(donutSlice, centersDistance)
	center = (int(donutSlice.xCenter), int(donutSlice.yCenter))
	newCenter = (int(newDonutSlice.xCenter), int(newDonutSlice.yCenter))
	theta0 = donutSlice.middleAngle - donutSlice.angleSpan / 2
	theta1 = donutSlice.middleAngle + donutSlice.angleSpan / 2
	bottomLeft = (int(donutSlice.xCenter + np.sin(theta0) * donutSlice.outRadius), \
		int(donutSlice.yCenter + np.cos(theta0) * donutSlice.outRadius))
	bottomRight = (int(donutSlice.xCenter + np.sin(theta1) * donutSlice.outRadius), \
		int(donutSlice.yCenter + np.cos(theta1) * donutSlice.outRadius))
	centerRadius = (newDonutSlice.inRadius + newDonutSlice.outRadius) / 2

	return DebugImageInfoParam(donutSlice=donutSlice, newDonutSlice=newDonutSlice, center=center, \
		newCenter=newCenter, bottomLeft=bottomLeft, bottomRight=bottomRight, centerRadius=centerRadius)


# Add debug lines on the source image
def addDebugInfoToImage(frame, debugImageInfoParam):
	donutSlice = debugImageInfoParam.donutSlice
	newDonutSlice = debugImageInfoParam.newDonutSlice

	cv2.circle(frame, debugImageInfoParam.center, int(donutSlice.inRadius), (255,0,255), 5)
	cv2.circle(frame, debugImageInfoParam.center, int(donutSlice.outRadius), (255,0,255), 5)
	cv2.circle(frame, debugImageInfoParam.center, int((donutSlice.inRadius + donutSlice.outRadius) / 2), (255,255,0), 5)

	cv2.line(frame, debugImageInfoParam.center, debugImageInfoParam.bottomLeft, (255,0,255), 5)
	cv2.line(frame, debugImageInfoParam.center, debugImageInfoParam.bottomRight, (255,0,255), 5)

	cv2.circle(frame, debugImageInfoParam.newCenter, int(newDonutSlice.inRadius), (255,0,122), 5)
	cv2.circle(frame, debugImageInfoParam.newCenter, int(newDonutSlice.outRadius), (255,0,122), 5)
	cv2.circle(frame, debugImageInfoParam.newCenter, int(debugImageInfoParam.centerRadius), (122,0,255), 5)

	cv2.line(frame, debugImageInfoParam.newCenter, (int(newDonutSlice.xCenter + np.sin(donutSlice.middleAngle) * newDonutSlice.outRadius), \
		int(newDonutSlice.yCenter + np.cos(donutSlice.middleAngle) * newDonutSlice.outRadius)), (255,0,122), 5)
	cv2.line(frame, debugImageInfoParam.newCenter, debugImageInfoParam.bottomLeft, (122,0,255), 5)
	cv2.line(frame, debugImageInfoParam.newCenter, debugImageInfoParam.bottomRight, (122,0,255), 5)


if __name__ == '__main__':
	main()
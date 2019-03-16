import time
from collections import namedtuple

import cv2
import numpy as np

import context
from src.utils.dewarping_helper import DewarpingHelper
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import FisheyeDewarping
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import DonutSlice

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
	outputWidth = 1280
	outputHeight = 720

	# Factor near zero means the dewarped image will follow the fisheye image curvature
	topDistorsionFactor = 0.08
	bottomDistorsionFactor = 0

	# Circles where the image cropping occurs (center of image to inRadius and outRadius to image borders will be cropped)
	inRadius = 400
	outRadius = 1400

	# 0 degree is bottom of source image, sets angular region to be dewarped
	middleAngle = 90
	angleSpan = 90

	donutSlice = DonutSlice(width / 2.0, height / 2.0, inRadius, outRadius, np.deg2rad(middleAngle), np.deg2rad(angleSpan))
	dewarpedImage = np.zeros((outputHeight, outputWidth, channels), dtype=np.uint8)
	dewarper = FisheyeDewarping()
	debugImageInfoParam = None

	if debug:
		print('DEBUG enabled')
		debugImageInfoParam = createDebugImageInfoParam(donutSlice, topDistorsionFactor)

	if dewarper.initialize(width, height, outputWidth, outputHeight, channels, True) == -1:
		print("Error during c++ lib initialization")
		return

	while True:
		success, frame = cam.read()

		if success:

			if debugImageInfoParam:
				addDebugInfoToImage(frame, debugImageInfoParam)

			dewarpingParameters = DewarpingHelper.getDewarpingParameters(donutSlice, topDistorsionFactor, bottomDistorsionFactor)
			dewarper.setDewarpingParameters(dewarpingParameters)
			dewarper.loadFisheyeImage(frame)
			dewarper.dewarpImage(dewarpedImage)
			cv2.imshow("img", dewarpedImage)

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
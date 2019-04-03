import time
from collections import namedtuple

import cv2
import numpy as np

import context
from src.utils.rect import Rect
from src.utils.dewarping_helper import DewarpingHelper
from src.utils.spherical_angles_converter import SphericalAnglesConverter
from src.utils.spherical_angles_rect import SphericalAnglesRect
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import FisheyeDewarping
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import DonutSlice
from src.app.services.videoprocessing.dewarping.interface.fisheye_dewarping import DewarpingParameters

DebugImageInfoParam = namedtuple('DebugImageInfoParam', 'donutSlice \
    newDonutSlice center newCenter bottomLeft bottomRight centerRadius')


def main():

	debug = False
	useCamera = False

	# Factor near zero means the dewarped image will follow the fisheye image curvature
	topDistorsionFactor = 0.08
	bottomDistorsionFactor = 0

	# Circles where the image cropping occurs (center of image to inRadius and outRadius to image borders will be cropped)
	inRadius = 400
	outRadius = 1400

	# 0 degree is bottom of source image, sets angular region to be dewarped
	middleAngle = 0
	angleSpan = 90

	if useCamera:
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

	else:
		# Put here the full apth to the image you want to use
		imagePath = ''

		if imagePath == '':
			print('!! You forgot to set camera mode, or to set a path to an image !!')
			return

		# Open image
		img = cv2.imread(imagePath, cv2.IMREAD_COLOR)

		# Get dimensions of image
		width = img.shape[1]
		height = img.shape[0]
		channels = img.shape[2]

	
	# Size the fisheye image window

	fisheyeOutputWidth = width // 4
	fisheyeOutputHeight = height // 4

	# 360 dewarping dewarping parameters

	dewarpingParametersList = getFaceDetectionDewarpingParameters(width, height, inRadius, outRadius, \
		angleSpan, topDistorsionFactor, bottomDistorsionFactor, 4)

	dewarpIndex = 0
	dewarpingParameters = dewarpingParametersList[dewarpIndex]

	# Face angle boundaries calculations (from hardcoded face rectangle)

	face = Rect(567.5, 163.5, 89, 71)
	fisheyeAngle = np.deg2rad(220)
	x1, y1, x2, y2 =  face.getBoundingRect()
	xCenter, yCenter = face.getPosition()
	fisheyeCenter = (width / 2, height / 2)

	outputWidth = roundDownToMultipleOf4(dewarpingParameters.dewarpWidth / 2)
	outputHeight = roundDownToMultipleOf4(dewarpingParameters.dewarpHeight / 2)	

	dewarpWidthFactor = dewarpingParameters.dewarpWidth / outputWidth
	dewarpHeightFactor = dewarpingParameters.dewarpHeight / outputHeight

	xNew1, yNew1, xNew2, yNew2 = (x1 * dewarpWidthFactor, y1 * dewarpHeightFactor, x2 * dewarpWidthFactor, y2 * dewarpHeightFactor)
	xNewCenter, yNewCenter = (xCenter * dewarpWidthFactor, yCenter* dewarpHeightFactor)

	if xCenter > outputWidth / 2:
		xMostTop = xNew1
		xMostBottom = xNew2
		yMostLeft = yNew2
		yMostRight = yNew1
	else:
		xMostTop = xNew2
		xMostBottom = xNew1
		yMostLeft = yNew1
		yMostRight = yNew2

	azimuthLeft = SphericalAnglesConverter.getAzimuthFromImage(xNew1, yMostLeft, \
		fisheyeAngle, fisheyeCenter, dewarpingParameters, True)
	azimuthRight = SphericalAnglesConverter.getAzimuthFromImage(xNew2, yMostRight, \
		fisheyeAngle, fisheyeCenter, dewarpingParameters, True)
	elevationTop = SphericalAnglesConverter.getElevationFromImage(xMostTop, yNew1, \
		fisheyeAngle, fisheyeCenter, dewarpingParameters)
	elevationBottom = SphericalAnglesConverter.getElevationFromImage(xMostBottom, yNew2, \
		fisheyeAngle, fisheyeCenter, dewarpingParameters)

	angleRect = SphericalAnglesRect(azimuthLeft, azimuthRight, elevationBottom, elevationTop)

	# Virtual camera dewarping parameters and output dimensions

	azimuth, elevation = angleRect.getMiddlePosition()
	azimuthSpan = angleRect.getAzimuthSpan()
	donutSliceVC = DonutSlice(width / 2, height / 2, inRadius, outRadius, azimuth, azimuthSpan)
	dewarpingParametersVC = DewarpingHelper.getDewarpingParameters(donutSliceVC, topDistorsionFactor, bottomDistorsionFactor)

	maxElevation = SphericalAnglesConverter.getElevationFromImage(dewarpingParametersVC.dewarpWidth / 2, 0, \
		fisheyeAngle, fisheyeCenter, dewarpingParametersVC)
	minElevation = SphericalAnglesConverter.getElevationFromImage(dewarpingParametersVC.dewarpWidth / 2, dewarpingParametersVC.dewarpHeight, \
		fisheyeAngle, fisheyeCenter, dewarpingParametersVC)

	deltaElevation = maxElevation - minElevation
	deltaElevationTop = maxElevation - elevationTop
	deltaElevationBottom = elevationBottom - minElevation

	dewarpingParametersVC.topOffset = (deltaElevationTop * dewarpingParametersVC.dewarpHeight) / deltaElevation
	dewarpingParametersVC.bottomOffset = (deltaElevationBottom * dewarpingParametersVC.dewarpHeight) / deltaElevation

	outputPortraitWidth = roundDownToMultipleOf4(dewarpingParametersVC.dewarpWidth)
	outputPortraitHeight = roundDownToMultipleOf4(dewarpingParametersVC.dewarpHeight - \
		dewarpingParametersVC.topOffset - dewarpingParametersVC.bottomOffset)

	# Initialization of dewarper and required buffers

	dewarper = FisheyeDewarping(width, height, channels)

	dewarpedImage = np.empty((outputHeight, outputWidth, channels), dtype=np.uint8)
	dewarpedPortrait = np.empty((outputPortraitHeight, outputPortraitWidth, channels), dtype=np.uint8)
	fisheyeImage = np.empty((fisheyeOutputHeight, fisheyeOutputWidth, channels), dtype=np.uint8)

	bufferId = dewarper.createRenderContext(outputWidth, outputHeight, channels)
	bufferPortraitId = dewarper.createRenderContext(outputPortraitWidth, outputPortraitHeight, channels)
	fisheyeBufferId = dewarper.createRenderContext(fisheyeOutputWidth, fisheyeOutputHeight, channels)

	dewarpedImages = {}
	dewarpedImages[bufferId] = dewarpedImage
	dewarpedImages[bufferPortraitId] = dewarpedPortrait
	dewarpedImages[fisheyeBufferId] = fisheyeImage

	# Initializatio of debug objects

	if debug:
		donutSlice = DonutSlice(width / 2.0, height / 2.0, inRadius, outRadius, np.deg2rad(middleAngle), np.deg2rad(angleSpan))
		debugImageInfoParam = createDebugImageInfoParam(donutSlice, topDistorsionFactor)
		debugImageInfoParamVC = createDebugImageInfoParam(donutSliceVC, topDistorsionFactor)

	# Rendering loop

	while True:

		if useCamera:
			success, frame = cam.read()

		else:
			success = True
			frame = img.copy()

		if success:
			if debug:
				addDebugInfoToImage(frame, debugImageInfoParam)
				addDebugInfoToImage(frame, debugImageInfoParamVC)
				addFaceDebugInfo(frame, face, outputWidth, outputHeight, dewarpingParameters)

			dewarper.loadFisheyeImage(frame)

			dewarper.queueDewarping(bufferId, dewarpingParameters, dewarpedImage)
			dewarper.queueDewarping(bufferPortraitId, dewarpingParametersVC, dewarpedPortrait)
			dewarper.queueRendering(fisheyeBufferId, fisheyeImage)

			buffer = dewarper.dewarpNextImage()
			while buffer != -1:

				# Add rectangle around the dwarped face
				if buffer == bufferId:
					cv2.rectangle(dewarpedImages[buffer], (x1, y1), (x2, y2), (255,0,255), 2)

				cv2.imshow("{buffer}".format(buffer=buffer), dewarpedImages[buffer])
				buffer = dewarper.dewarpNextImage()

		# Keyboard inputs, dewarping parameters are currently not recalculated every frame
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

def getFaceDetectionDewarpingParameters(imageWidth, imageHeight, inRadius, outRadius, angleSpan, topDistorsionFactor, bottomDistorsionFactor, dewarpCount):
	dewarpingParametersList = []
	donutSlice = DonutSlice(imageWidth / 2, imageHeight / 2, inRadius, outRadius, np.deg2rad(0), np.deg2rad(angleSpan))

	for i in range(0, dewarpCount):
		dewarpingParameters = DewarpingHelper.getDewarpingParameters(donutSlice, topDistorsionFactor, bottomDistorsionFactor)
		dewarpingParameters.topOffset = 0
		dewarpingParameters.bottomOffset = 0
		dewarpingParametersList.append(dewarpingParameters)
		donutSlice.middleAngle = (donutSlice.middleAngle + np.deg2rad(360 / dewarpCount)) % (2 * np.pi)

	return dewarpingParametersList


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


def addFaceDebugInfo(frame, face, outputWidth, outputHeight, dewarpingParameters):

	x1, y1, x2, y2 =  face.getBoundingRect()
	xCenter, yCenter = face.getPosition()

	dewarpWidthFactor = dewarpingParameters.dewarpWidth / outputWidth
	dewarpHeightFactor = dewarpingParameters.dewarpHeight / outputHeight

	xNew1, yNew1, xNew2, yNew2 = (x1 * dewarpWidthFactor, y1 * dewarpHeightFactor, x2 * dewarpWidthFactor, y2 * dewarpHeightFactor)
	xNewCenter, yNewCenter = (xCenter * dewarpWidthFactor, yCenter* dewarpHeightFactor)

	x1SrcPixel, y1SrcPixel = DewarpingHelper.getSourcePixelFromDewarpedImage(xNew1, yNew1, dewarpingParameters)
	x2SrcPixel, y2SrcPixel = DewarpingHelper.getSourcePixelFromDewarpedImage(xNew2, yNew2, dewarpingParameters)
	x3SrcPixel, y3SrcPixel = DewarpingHelper.getSourcePixelFromDewarpedImage(xNew1, yNew2, dewarpingParameters)
	x4SrcPixel, y4SrcPixel = DewarpingHelper.getSourcePixelFromDewarpedImage(xNew2, yNew1, dewarpingParameters)

	xLeftSrcPixel, yLeftSrcPixel = DewarpingHelper.getSourcePixelFromDewarpedImage(xNew1, yNewCenter, dewarpingParameters)
	xRightSrcPixel, yRightSrcPixel = DewarpingHelper.getSourcePixelFromDewarpedImage(xNew2, yNewCenter, dewarpingParameters)
	xTopSrcPixel, yTopSrcPixel = DewarpingHelper.getSourcePixelFromDewarpedImage(xNewCenter, yNew1, dewarpingParameters)
	xBottomSrcPixel, yBottomSrcPixel = DewarpingHelper.getSourcePixelFromDewarpedImage(xNewCenter, yNew2, dewarpingParameters)
	xCenterSrcPixel, yCenterSrcPixel = DewarpingHelper.getSourcePixelFromDewarpedImage(xNewCenter, yNewCenter, dewarpingParameters)

	cv2.circle(frame, (int(x1SrcPixel), int(y1SrcPixel)), 10, (255,0,255), 5)
	cv2.circle(frame, (int(x2SrcPixel), int(y2SrcPixel)), 10, (255,0,255), 5)
	cv2.circle(frame, (int(x3SrcPixel), int(y3SrcPixel)), 10, (255,0,255), 5)
	cv2.circle(frame, (int(x4SrcPixel), int(y4SrcPixel)), 10, (255,0,255), 5)
	
	cv2.circle(frame, (int(xLeftSrcPixel), int(yLeftSrcPixel)), 10, (255,0,0), 5)
	cv2.circle(frame, (int(xRightSrcPixel), int(yRightSrcPixel)), 10, (255,0,0), 5)
	cv2.circle(frame, (int(xTopSrcPixel), int(yTopSrcPixel)), 10, (255,0,0), 5)
	cv2.circle(frame, (int(xBottomSrcPixel), int(yBottomSrcPixel)), 10, (255,0,0), 5)
	cv2.circle(frame, (int(xCenterSrcPixel), int(yCenterSrcPixel)), 10, (255,0,0), 5)


def roundDownToMultipleOf4(value):
	return int(value) - int(value) % 4

if __name__ == '__main__':
	main()
from enum import Enum, unique
from math import degrees
import numpy as np
import random
import time

from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication, QLabel, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import pyqtSlot, QTimer
import pyqtgraph as pg

from src.app.gui.conference_ui import Ui_Conference
from src.app.controllers.conference_controller import ConferenceController

from src.app.services.videoprocessing.virtualcamera.virtual_camera_displayer import VirtualCameraDisplayer


@unique
class BtnOdasLabels(Enum):
    START_ODAS = 'Start ODAS'
    STOP_ODAS = 'Stop ODAS'


@unique
class BtnVideoLabels(Enum):
    START_VIDEO = 'Start Video'
    STOP_VIDEO = 'Stop Video'


@unique
class FontSizes(Enum):
    SMALL_SIZE = 9
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12


class Conference(QWidget, Ui_Conference):

    def __init__(self, odasserver, parent=None):
        super(Conference, self).__init__(parent)
        self.setupUi(self)
        self.conferenceController = ConferenceController(odasserver)

        self.virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)

        # positions graphs initialization
        self.azimuthGraph = Graph(self.soundPositionsLayout, curvesNumber=4, maxLength=500, yMax=370, title='Azimuth Positions')
        self.elevationGraph = Graph(self.soundPositionsLayout, curvesNumber=4, maxLength=500, yMax=90, title='Elevation Positions')

        self.__isVideoSignalsConnected = False
        self.__isOdasSignalsConnected = False

        # Qt signal slots
        self.btnStartStopOdas.clicked.connect(self.btnStartStopOdasClicked)
        self.btnStartStopVideo.clicked.connect(self.btnStartStopVideoClicked)

        self.conferenceController.signalException.connect(self.exceptionReceived)
        self.conferenceController.signalHumanSourcesDetected.connect(self.showHumanSources)

        self.soundSources = []


    @pyqtSlot()
    def btnStartStopOdasClicked(self):
        self.btnStartStopOdas.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopOdas.text() == BtnOdasLabels.START_ODAS.value:
            self.conferenceController.signalAudioPositions.connect(self.positionDataReceived)
            self.conferenceController.signalOdasState.connect(self.odasStateChanged)
            self.__isOdasSignalsConnected = True
            self.conferenceController.startOdasLive(odasPath=self.window().getSetting('odasPath'), micConfigPath=self.window().getSetting('micConfigPath'))
        else:
            self.conferenceController.stopOdasLive()
            self.azimuthGraph.resetData()
            self.elevationGraph.resetData()


    @pyqtSlot()
    def btnStartStopVideoClicked(self):
        self.btnStartStopVideo.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopVideo.text() == BtnVideoLabels.START_VIDEO.value:
            self.conferenceController.signalVideoProcessorState.connect(self.videoProcessorStateChanged)
            self.conferenceController.signalVirtualCamerasReceived.connect(self.updateVirtualCamerasDispay)
            self.__isVideoSignalsConnected = True
            self.conferenceController.startVideoProcessor(self.window().getSetting('cameraConfigPath'), self.window().getSetting('faceDetection'))
        else:
            self.conferenceController.stopVideoProcessor()


    @pyqtSlot(object)
    def positionDataReceived(self, values):
        azimuthNewData = []
        elevationNewData = []
        for angles in values:
            azimuthNewData.append(float(np.rad2deg(angles['azimuth'])))
            elevationNewData.append(float(np.rad2deg(angles['elevation'])))
        self.azimuthGraph.addData(azimuthNewData)
        self.elevationGraph.addData(elevationNewData)

        self.soundSources = values


    @pyqtSlot(object)        
    def updateVirtualCamerasDispay(self, virtualCameraImages):
        self.virtualCameraDisplayer.updateDisplay(virtualCameraImages)


    @pyqtSlot(bool)
    def odasStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopOdas.setText(BtnOdasLabels.STOP_ODAS.value)
            self.azimuthGraph.timer.start(500)
            self.elevationGraph.timer.start(500)
        
        else:
            self.btnStartStopOdas.setText(BtnOdasLabels.START_ODAS.value)
            self.conferenceController.signalAudioPositions.disconnect(self.positionDataReceived)
            self.conferenceController.signalOdasState.disconnect(self.odasStateChanged)
            self.__isOdasSignalsConnected = False
            self.azimuthGraph.timer.stop()
            self.elevationGraph.timer.stop()

        self.btnStartStopOdas.setDisabled(False)

    
    @pyqtSlot(bool)
    def videoProcessorStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopVideo.setText(BtnVideoLabels.STOP_VIDEO.value)
            self.virtualCameraDisplayer.startDisplaying()
            
        else:
            self.btnStartStopVideo.setText(BtnVideoLabels.START_VIDEO.value)
            self.virtualCameraDisplayer.stopDisplaying()
            self.conferenceController.signalVideoProcessorState.disconnect(self.videoProcessorStateChanged)
            self.conferenceController.signalVirtualCamerasReceived.disconnect(self.updateVirtualCamerasDispay)
            self.__isVideoSignalsConnected = False

        self.btnStartStopVideo.setDisabled(False)


    @pyqtSlot(Exception)
    def exceptionReceived(self, e):
        if self.__isVideoSignalsConnected:
            self.conferenceController.signalVideoProcessorState.disconnect(self.videoProcessorStateChanged)
            self.conferenceController.signalVirtualCamerasReceived.disconnect(self.updateVirtualCamerasDispay)
        
        if self.__isOdasSignalsConnected:
            self.conferenceController.signalAudioPositions.disconnect(self.positionDataReceived)
            self.conferenceController.signalOdasState.disconnect(self.odasStateChanged)
        
        self.window().emitToExceptionsManager(e)


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.conferenceController.stopOdasServer()
            self.conferenceController.stopVideoProcessor()
            event.accept()
    

    def showHumanSources(self, humanSources):
        for index, source in enumerate(self.soundSources):
            if index in humanSources:
                self.__setSourceBackgroundColor(index, 'yellow')
            else:
                self.__setSourceBackgroundColor(index, 'transparent')


    def __setSourceBackgroundColor(self, index, color):
        pass


class Graph(QWidget):

    timer = QTimer()
    maxLength = None
    startIndex = 0

    def __init__(self, layoutDisplay, curvesNumber, yMax, maxLength=None, title='', parent=None):
        super(Graph, self).__init__(parent)
        self.maxLength = maxLength
        self.curvesNumber = curvesNumber
        self.lines = []
        self.linesColor = []
        self.data = []

        self.plot = pg.PlotWidget(title=title)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.plot.setMaximumHeight(200)
        self.plot.setLabel('left', 'Angle', 'degree')
        self.plot.setLabel('bottom', 'Sample')
        self.plot.showGrid(x=True, y=True)
        self.plot.hideButtons()
        self.plot.setRange(yRange=[-1, yMax], xRange=[0, self.maxLength])
        self.plot.setLimits(xMin=0, xMax=self.maxLength, yMin=-1, yMax=yMax)
        self.plot.setBackground([255, 255, 255])
        layoutDisplay.addWidget(self.plot)

        self.randomColor = lambda: random.randint(100,255)  
        self.computeInitialGraph()

        self.timer.timeout.connect(self.updateGraph)


    def addData(self, samples):
        if samples:
            for i, sample in enumerate(samples):
                self.data[i].append(sample)


    def resetData(self):
        self.data = []
        for i in range(0, self.curvesNumber):
            self.data.append([])


    def computeInitialGraph(self):
        for i in range(0, self.curvesNumber):
            lineObj = self.plot.plot()
            self.lines.append(lineObj)
            
            # Choose a random color for each line.
            # Make sure there is no lines with the same color.
            color = [self.randomColor(), self.randomColor(), self.randomColor()]
            while color in self.linesColor:
                color = [self.randomColor(), self.randomColor(), self.randomColor()]

            self.linesColor.append(color)

            self.data.append([])


    def updateGraph(self):
        xData = []
        yData = []

        for index, lineData in enumerate(self.data):
            if self.maxLength and len(lineData) > self.maxLength:
                newDataLength = len(lineData) - self.maxLength
                yData = lineData[self.startIndex + newDataLength : len(lineData)]
                self.data[index] = yData
                self.startIndex = 0

            else:
                yData = lineData

            xData = np.arange(0, len(yData))
            self.lines[index].setData(x=xData, y=yData, pen=pg.mkPen(self.linesColor[index], width=2))


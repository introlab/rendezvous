from enum import Enum, unique
from math import degrees
import numpy as np
import random
import time

import pyqtgraph as pg

from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QSizePolicy, QVBoxLayout, QWidget

from src.app.application_container import ApplicationContainer
from src.app.controllers.conference_controller import ConferenceController
from src.app.gui.conference_ui import Ui_Conference
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

    def __init__(self, parent=None):
        super(Conference, self).__init__(parent)
        self.setupUi(self)

        self.__virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)

        self.__conferenceController = ConferenceController(self.virtualCameraFrame)
        self.__conferenceController.signalAudioPositions.connect(self.__positionDataReceived)
        self.__conferenceController.signalOdasState.connect(self.__odasStateChanged)
        self.__conferenceController.signalVideoProcessorState.connect(self.__videoProcessorStateChanged)
        self.__conferenceController.signalVirtualCamerasReceived.connect(self.__updateVirtualCamerasDisplay)
        self.__conferenceController.signalException.connect(self.__exceptionReceived)

        # positions graphs initialization
        self.__azimuthGraph = Graph(self.soundPositionsLayout, curvesNumber=4, maxLength=500, yMax=370, title='Azimuth Positions')
        self.__elevationGraph = Graph(self.soundPositionsLayout, curvesNumber=4, maxLength=500, yMax=90, title='Elevation Positions')

        # Qt signal slots
        self.btnStartStopOdas.clicked.connect(self.__btnStartStopOdasClicked)
        self.btnStartStopVideo.clicked.connect(self.__btnStartStopVideoClicked)

        self.soundSources = []


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.__conferenceController.close()
            event.accept()


    @pyqtSlot()
    def __btnStartStopOdasClicked(self):
        self.btnStartStopOdas.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopOdas.text() == BtnOdasLabels.START_ODAS.value:
            self.__conferenceController.startOdasLive(ApplicationContainer.settings().getValue('odasPath'),
                                                    ApplicationContainer.settings().getValue('micConfigPath'))
        else:
            self.__conferenceController.stopOdasLive()
            self.__azimuthGraph.resetData()
            self.__elevationGraph.resetData()


    @pyqtSlot()
    def __btnStartStopVideoClicked(self):
        self.btnStartStopVideo.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopVideo.text() == BtnVideoLabels.START_VIDEO.value:
            self.__conferenceController.startVideoProcessor(ApplicationContainer.settings().getValue('cameraConfigPath'),
                                                          ApplicationContainer.settings().getValue('faceDetection'))
        else:
            self.__conferenceController.stopVideoProcessor()


    @pyqtSlot(object)
    def __positionDataReceived(self, values):
        azimuthNewData = []
        elevationNewData = []
        for angles in values:
            azimuthNewData.append(float(np.rad2deg(angles['azimuth'])))
            elevationNewData.append(float(np.rad2deg(angles['elevation'])))
        self.__azimuthGraph.addData(azimuthNewData)
        self.__elevationGraph.addData(elevationNewData)

        self.soundSources = values


    @pyqtSlot(object)
    def __updateVirtualCamerasDisplay(self, virtualCameraImage):
        self.__virtualCameraDisplayer.updateDisplay(virtualCameraImage)


    @pyqtSlot(bool)
    def __odasStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopOdas.setText(BtnOdasLabels.STOP_ODAS.value)
            self.__azimuthGraph.timer.start(500)
            self.__elevationGraph.timer.start(500)
        
        else:
            self.btnStartStopOdas.setText(BtnOdasLabels.START_ODAS.value)
            self.__azimuthGraph.timer.stop()
            self.__elevationGraph.timer.stop()

        self.btnStartStopOdas.setDisabled(False)

    
    @pyqtSlot(bool)
    def __videoProcessorStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopVideo.setText(BtnVideoLabels.STOP_VIDEO.value)
            self.__virtualCameraDisplayer.startDisplaying()
            
        else:
            self.btnStartStopVideo.setText(BtnVideoLabels.START_VIDEO.value)
            self.__virtualCameraDisplayer.stopDisplaying()

        self.btnStartStopVideo.setDisabled(False)


    @pyqtSlot(Exception)
    def __exceptionReceived(self, e):
        ApplicationContainer.exceptions().show(e)


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


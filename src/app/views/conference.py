from enum import Enum, unique
from math import degrees
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication, QLabel, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import pyqtSlot, QTimer

from src.app.gui.conference_ui import Ui_Conference
from src.app.controllers.conference_controller import ConferenceController

from src.app.services.videoprocessing.virtualcamera.virtual_camera_displayer import VirtualCameraDisplayer


@unique
class BtnAudioRecordLabels(Enum):
    START_RECORDING = 'Start Audio Recording'
    STOP_RECORDING = 'Stop Audio Recording'


@unique
class BtnOdasLabels(Enum):
    START_ODAS = 'Start ODAS'
    STOP_ODAS = 'Stop ODAS'


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
        self.conferenceController = ConferenceController()
        self.outputFolder.setText(self.window().getSetting('outputFolder'))

        self.virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)

        self.btnStartStopAudioRecord.setDisabled(True)

        # positions graphs initialization
        plt.rc('font', size=FontSizes.SMALL_SIZE.value)          # controls default text sizes
        plt.rc('axes', titlesize=FontSizes.SMALL_SIZE.value)     # fontsize of the axes title
        plt.rc('axes', labelsize=FontSizes.MEDIUM_SIZE.value)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=FontSizes.SMALL_SIZE.value)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=FontSizes.SMALL_SIZE.value)    # fontsize of the tick labels
        plt.rc('legend', fontsize=FontSizes.SMALL_SIZE.value)    # legend fontsize
        plt.rc('figure', titlesize=FontSizes.BIGGER_SIZE.value)  # fontsize of the figure title

        self.azimuthGraph = Graph(None, width=1, height=0.5, dpi=100, title='Azimuth Positions')
        self.elevationGraph = Graph(None, width=1, height=0.5, dpi=100, title='Elevation Positions')
        self.soundPositionsVerticalLayout.addWidget(self.azimuthGraph)
        self.soundPositionsVerticalLayout.addWidget(self.elevationGraph)

        # Qt signal slots
        self.btnSelectOutputFolder.clicked.connect(self.selectOutputFolder)
        self.btnStartStopOdas.clicked.connect(self.btnStartStopOdasClicked)
        self.btnStartStopVideo.clicked.connect(self.btnStartStopVideoClicked)
        self.btnStartStopAudioRecord.clicked.connect(self.btnStartStopAudioRecordClicked)

        self.conferenceController.signalAudioPositions.connect(self.positionDataReceived)
        self.conferenceController.signalOdasState.connect(self.odasStateChanged)
        self.conferenceController.signalRecordingState.connect(self.recordingStateChanged)
        self.conferenceController.signalVideoProcessorState.connect(self.videoProcessorStateChanged)
        self.conferenceController.signalVirtualCamerasReceived.connect(self.updateVirtualCamerasDispay)
        self.conferenceController.signalHumanSourcesDetected.connect(self.showHumanSources)
        self.conferenceController.signalException.connect(self.exceptionReceived)


    @pyqtSlot()
    def selectOutputFolder(self):
        try:
            outputFolder = QFileDialog.getExistingDirectory(
                parent=self, 
                caption='Select Output Directory', 
                directory=self.window().rootDirectory,
                options=QFileDialog.DontUseNativeDialog
            )
            if outputFolder:
                self.outputFolder.setText(outputFolder)
                self.window().setSetting('outputFolder', outputFolder)

        except Exception as e:
            self.window().emitToExceptionManager(e)


    @pyqtSlot()
    def btnStartStopOdasClicked(self):
        self.btnStartStopOdas.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopOdas.text() == BtnOdasLabels.START_ODAS.value:
            self.conferenceController.startOdasLive(odasPath=self.window().getSetting('odasPath'), micConfigPath=self.window().getSetting('micConfigPath'))
        else:
            self.conferenceController.stopOdasLive()


    @pyqtSlot()
    def btnStartStopVideoClicked(self):
        self.btnStartStopVideo.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopVideo.text() == BtnVideoLabels.START_VIDEO.value:
            self.conferenceController.startVideoProcessor(self.window().getSetting('cameraConfigPath'), self.window().getSetting('faceDetection'))
        else:
            self.conferenceController.stopVideoProcessor()


    @pyqtSlot()
    def btnStartStopAudioRecordClicked(self):
        if not self.outputFolder.text():
            self.window().emitToExceptionsManager(Exception('output folder cannot be empty'))

        self.btnStartStopAudioRecord.setDisabled(True)
        QApplication.processEvents()

        if self.btnStartStopAudioRecord.text() == BtnAudioRecordLabels.START_RECORDING.value:
            self.conferenceController.startAudioRecording(self.outputFolder.text())

        else:
            self.conferenceController.saveAudioRecording()


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
    def recordingStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopAudioRecord.setText(BtnAudioRecordLabels.STOP_RECORDING.value)
        else:
            self.btnStartStopAudioRecord.setText(BtnAudioRecordLabels.START_RECORDING.value)

        self.btnStartStopAudioRecord.setDisabled(False)


    @pyqtSlot(bool)
    def odasStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopOdas.setText(BtnOdasLabels.STOP_ODAS.value)
            self.btnStartStopAudioRecord.setDisabled(False)
        else:
            self.btnStartStopOdas.setText(BtnOdasLabels.START_ODAS.value)
            self.btnStartStopAudioRecord.setText(BtnAudioRecordLabels.START_RECORDING.value)
            self.conferenceController.saveAudioRecording()
            self.btnStartStopAudioRecord.setDisabled(True)

        self.btnStartStopOdas.setDisabled(False)

    
    @pyqtSlot(bool)
    def videoProcessorStateChanged(self, isRunning):
        if isRunning:
            self.btnStartStopVideo.setText(BtnVideoLabels.STOP_VIDEO.value)
            self.virtualCameraDisplayer.startDisplaying()
            
        else:
            self.btnStartStopVideo.setText(BtnVideoLabels.START_VIDEO.value)
            self.virtualCameraDisplayer.stopDisplaying()

        self.btnStartStopVideo.setDisabled(False)


    @pyqtSlot(Exception)
    def exceptionReceived(self, e):
        self.window().emitToExceptionsManager(e)


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            self.conferenceController.stopVideoProcessor()
            self.conferenceController.stopAudioRecording()
            self.conferenceController.stopOdasServer()
            event.accept()
    

    def showHumanSources(self, humanSources):
        for index, source in enumerate(self.soundSources):
            if index in humanSources:
                self.__setSourceBackgroundColor(index, 'yellow')
            else:
                self.__setSourceBackgroundColor(index, 'transparent')


    def __setSourceBackgroundColor(self, index, color):
        if index == 0:
            self.source1.setStyleSheet('background-color: %s' % color)
        elif index == 1:
            self.source2.setStyleSheet('background-color: %s' % color)
        elif index == 2:
            self.source3.setStyleSheet('background-color: %s' % color)
        elif index == 3:
            self.source4.setStyleSheet('background-color: %s' % color)


class Canvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, title=None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.title = title
        if self.title:
            self.axes.set_title(title)

        self.data = []

        self.computeInitialFigure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


    def computeInitialFigure(self):
        pass


class Graph(Canvas): 
	
    def __init__(self, *args, **kwargs):
        Canvas.__init__(self, *args, **kwargs)
        timer = QTimer(self)
        timer.timeout.connect(self.updateFigure)
        timer.start(500)


    def computeInitialFigure(self):
        self.axes.plot([], self.data)


    def addData(self, data):
        if data:
            self.data.append(data)


    def updateFigure(self):
        self.axes.cla()
        yData = self.data
        xData = range(0, len(self.data))
        self.axes.plot(xData, yData)
        self.axes.set_title(self.title)

        self.draw()


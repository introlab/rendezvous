from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSlot

from src.app.gui.odas_live_ui import Ui_OdasLive
from src.app.virtualcamera.virtual_camera_displayer import VirtualCameraDisplayer
from src.app.virtualcamera.virtual_camera_manager import VirtualCameraManager


class OdasLive(QWidget, Ui_OdasLive):
    def __init__(self, odasStream, videoProcessor, parent=None):
        super(OdasLive, self).__init__(parent)
        self.setupUi(self)
        self.odasStream = odasStream
        self.videoProcessor = videoProcessor

        self.virtualCameraManager = VirtualCameraManager()
        self.virtualCameraDisplayer = VirtualCameraDisplayer(self.virtualCameraFrame)

        # Qt signal slots
        self.odasStream.signalOdasData.connect(self.odasDataReveived)
        self.videoProcessor.signalFrameData.connect(self.imageReceived)

        self.btnStartOdas.clicked.connect(self.btnStartOdasClicked)
        self.btnStopOdas.clicked.connect(self.btnStopOdasClicked)
        self.btnStartVideo.clicked.connect(self.btnStartVideoClicked)
        self.btnStopVideo.clicked.connect(self.btnStopVideoClicked)
    

    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        self.stopVideoProcessor()
        self.stopOdas()
        event.accept()


    def startOdas(self):
        if self.odasStream and not self.odasStream.isRunning:
            self.odasStream.start()


    def stopOdas(self):
        if self.odasStream and self.odasStream.isRunning:
            self.odasStream.stop()


    def startVideoProcessor(self):
        if self.videoProcessor and not self.videoProcessor.isRunning:
            self.videoProcessor.start()


    def stopVideoProcessor(self):
        if self.videoProcessor and self.videoProcessor.isRunning:
            self.videoProcessor.stop()


    @pyqtSlot()
    def btnStartOdasClicked(self):
        self.startOdas()
        

    @pyqtSlot()
    def btnStopOdasClicked(self):
        self.stopOdas()


    @pyqtSlot()
    def btnStartVideoClicked(self):
        self.startVideoProcessor()
        

    @pyqtSlot()
    def btnStopVideoClicked(self):
        self.stopVideoProcessor()


    @pyqtSlot(object)
    def odasDataReveived(self, values):
        self.source1AzimuthValueLabel.setText('%.5f' % values[0]['azimuth'])
        self.source2AzimuthValueLabel.setText('%.5f' % values[1]['azimuth'])
        self.source3AzimuthValueLabel.setText('%.5f' % values[2]['azimuth'])
        self.source4AzimuthValueLabel.setText('%.5f' % values[3]['azimuth'])

        self.source1ElevationValueLabel.setText('%.5f' % values[0]['elevation'])
        self.source2ElevationValueLabel.setText('%.5f' % values[1]['elevation'])
        self.source3ElevationValueLabel.setText('%.5f' % values[2]['elevation'])
        self.source4ElevationValueLabel.setText('%.5f' % values[3]['elevation'])


    @pyqtSlot(object, object)
    def imageReceived(self, image, faces):
        imageHeight, imageWidth, colors = image.shape

        for face in faces:
            self.virtualCameraManager.addOrUpdateVirtualCamera(face, imageWidth, imageHeight)

        self.virtualCameraDisplayer.updateDisplay(image, self.virtualCameraManager.virtualCameras)


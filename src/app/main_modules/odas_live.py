from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSlot

from src.app.gui.odas_live_ui import Ui_OdasLive

class OdasLive(QWidget, Ui_OdasLive):
    def __init__(self, odasStream, parent=None):
        super(OdasLive, self).__init__(parent)
        self.setupUi(self)
        self.odasStream = odasStream

        # Qt signal slots
        self.odasStream.signalOdasData.connect(self.odasDataReveived)
        self.btnStartOdas.clicked.connect(self.btnStartOdasClicked)
        self.btnStopOdas.clicked.connect(self.btnStopOdasClicked)
    

    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        self.stopOdas()
        event.accept()


    def startOdas(self):
        if self.odasStream and not self.odasStream.isRunning:
            self.odasStream.start()


    def stopOdas(self):
        if self.odasStream and self.odasStream.isRunning:
                self.odasStream.stop()


    @pyqtSlot()
    def btnStartOdasClicked(self):
        self.startOdas()


    @pyqtSlot()
    def btnStopOdasClicked(self):
        self.stopOdas()


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

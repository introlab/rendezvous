import sys
import re
import os

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSlot

import context
from src.prototype.gui.mainwidow_ui import Ui_MainWindow
from src.prototype.odasstream.odas_stream import OdasStream
from src.utils.file_helper import FileHelper
from src.prototype.argsparser.args_parser import ArgsParser



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, odasStream, parent=None):
        super(MainWindow, self).__init__(parent)
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


def main():
    parser = ArgsParser()
    args = parser.args

    # Read config file to get sample rate for while True sleepTime
    line = FileHelper.getLineFromFile(args.configPath, 'fS')

    # Extract the sample rate from the string and convert to an Integer
    sampleRate = int(re.sub('[^0-9]', '', line.split('=')[1]))
    sleepTime = 1 / sampleRate

    stream = OdasStream(args.odasPath, args.configPath, sleepTime)

    app = QApplication(sys.argv)
    main_window = MainWindow(stream)
    main_window.show()
    exit(app.exec_())


if __name__ == '__main__':
    main()


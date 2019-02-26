import sys
import re
import os
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSlot
from gui.mainwidow_ui import Ui_MainWindow
from OdasStream.odas_stream import OdasStream
from FileHelper.file_helper import FileHelper
from ArgsParser.args_parser import ArgsParser


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, odasStream, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.odasStream = odasStream

        # signal slots
        self.odasStream.signalOdasData.connect(self.odasDataReveived)
        self.btnStartOdas.clicked.connect(self.btnStartOdasClicked)
        self.btnStopOdas.clicked.connect(self.btnStopOdasClicked)


    def closeEvent(self, event):
        self.stopOdas()
        event.accept()


    def startOdas(self):
        if self.odasStream and not self.odasStream.isRunning:
            self.odasStream.start()


    def stopOdas(self):
        if self.odasStream and self.odasStream.odasProcess:
            if self.odasStream.isRunning:
                self.odasStream.stop()


    @pyqtSlot()
    def btnStartOdasClicked(self):
        self.startOdas()


    @pyqtSlot()
    def btnStopOdasClicked(self):
        self.stopOdas()


    @pyqtSlot(object)
    def odasDataReveived(self, values):
        self.source1AzimuthValue.setText('%.5f' % values[0]['azimuth'])
        self.source2AzimuthValue.setText('%.5f' % values[1]['azimuth'])
        self.source3AzimuthValue.setText('%.5f' % values[2]['azimuth'])
        self.source4AzimuthValue.setText('%.5f' % values[3]['azimuth'])

        self.source1ElevationValue.setText('%.5f' % values[0]['elevation'])
        self.source2ElevationValue.setText('%.5f' % values[1]['elevation'])
        self.source3ElevationValue.setText('%.5f' % values[2]['elevation'])
        self.source4ElevationValue.setText('%.5f' % values[3]['elevation'])


def main():
    # get terminal arguments.
    parser = ArgsParser()
    args = parser.args

    # read config file to get sample rate for while True sleepTime
    line = FileHelper.getLineFromFile(args.configPath, 'fS')

    # extract the sample rate from the string and convert to an Integer
    sampleRate = int(re.sub('[^0-9]', '', line.split('=')[1]))
    sleepTime = 1 / sampleRate

    stream = OdasStream(args.odasPath, args.configPath, sleepTime)

    app = QApplication(sys.argv)
    main_window = MainWindow(stream)
    main_window.show()
    exit(app.exec_())


if __name__ == '__main__':
    main()


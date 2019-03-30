import traceback

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox


class Exceptions(QObject):

    signalException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(Exceptions, self).__init__(parent)
        self.signalException.connect(self.exceptionHandling)


    def exceptionHandling(self, e):
        if e:
            print('Exception : ', e)
            traceback.print_tb(e.__traceback__)
            dlg = QMessageBox()
            dlg.setIcon(QMessageBox.Warning)
            dlg.setText(str(e))
            dlg.setWindowTitle('Warning')
            dlg.setStandardButtons(QMessageBox.Ok)
            dlg.exec_()

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

class ExceptionManager(QObject):
    
    signalException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(ExceptionManager, self).__init__(parent)
        self.signalException.connect(self.ExceptionHandling)

    def ExceptionHandling(self, e):
        dlg = QMessageBox()
        dlg.setIcon(QMessageBox.Warning)
        dlg.setText(str(e))
        dlg.setWindowTitle('Warning')
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.exec_()

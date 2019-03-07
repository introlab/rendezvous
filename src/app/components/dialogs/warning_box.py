from PyQt5.QtWidgets import QMessageBox

class WarningBox():
    def __init__(self, msg):
        dlg = QMessageBox()
        dlg.setIcon(QMessageBox.Warning)
        dlg.setText(str(msg))
        dlg.setWindowTitle('Warning')
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.exec_()

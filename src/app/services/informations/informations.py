from PyQt5.QtWidgets import QMessageBox


class Informations(object):

    @staticmethod
    def show(message):
        if message:
            dlg = QMessageBox()
            dlg.setIcon(QMessageBox.Information)
            dlg.setText(str(message))
            dlg.setWindowTitle('Information')
            dlg.setStandardButtons(QMessageBox.Ok)
            dlg.exec_()


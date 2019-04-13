from PyQt5.QtWidgets import QMessageBox

from src.utils.exception_helper import ExceptionHelper


class Exceptions(object):

    @staticmethod
    def show(error):
        if error:
            ExceptionHelper.printStackTrace(error)
            dlg = QMessageBox()
            dlg.setIcon(QMessageBox.Warning)
            dlg.setText(str(error))
            dlg.setWindowTitle('Warning')
            dlg.setStandardButtons(QMessageBox.Ok)
            dlg.exec_()


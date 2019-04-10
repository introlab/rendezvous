import traceback

from PyQt5.QtWidgets import QMessageBox


class Exceptions(object):

    @staticmethod
    def show(error):
        if error:
            print('Exception : ', error)
            traceback.print_tb(error.__traceback__)
            dlg = QMessageBox()
            dlg.setIcon(QMessageBox.Warning)
            dlg.setText(str(error))
            dlg.setWindowTitle('Warning')
            dlg.setStandardButtons(QMessageBox.Ok)
            dlg.exec_()

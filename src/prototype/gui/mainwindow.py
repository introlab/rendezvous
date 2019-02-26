import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from gui.mainwidow_ui import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

    
    def btnStartOdasClicked(self):
        print("Hello")
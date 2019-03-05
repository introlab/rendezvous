from PyQt5.QtWidgets import QMainWindow

import context
from src.app.gui.mainwindow_ui import Ui_MainWindow
from src.app.main_modules.odas_live import OdasLive

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, odasStream, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.odasLiveTab = OdasLive(odasStream, parent=self)   
        
        self.tabWidget.addTab(self.odasLiveTab, "Odas Live") 
from PyQt5.QtWidgets import QMainWindow

from src.app.gui.main_window_ui import Ui_MainWindow
from src.app.main_modules.odas_live import OdasLive
from src.app.main_modules.speech_to_text import SpeechToText

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, odasStream, videoProcessor, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.odasLiveTab = OdasLive(odasStream, videoProcessor, parent=self)   
        self.tabWidget.addTab(self.odasLiveTab, 'Odas Live')

        self.speechToTextTab = SpeechToText(parent=self)   
        self.tabWidget.addTab(self.speechToTextTab, 'Speech to Text')


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        self.odasLiveTab.closeEvent(event)
        self.speechToTextTab.closeEvent(event)
        event.accept()
        
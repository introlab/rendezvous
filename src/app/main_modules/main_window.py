from PyQt5.QtWidgets import QMainWindow

from src.app.gui.main_window_ui import Ui_MainWindow
from src.app.main_modules.exception_manager import ExceptionManager
from src.app.main_modules.odas_live import OdasLive
from src.app.main_modules.speech_to_text import SpeechToText
from src.app.main_modules.settings import SettingsDialog


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # Exception manager.
        self.exceptionManager = ExceptionManager()

        # Top options.
        self.actionSettings.triggered.connect(self.actionSettingsClicked)

        # Tabs of the main layout.
        self.odasLiveTab = OdasLive(parent=self)   
        self.tabWidget.addTab(self.odasLiveTab, 'Odas Live')

        self.speechToTextTab = SpeechToText(parent=self)   
        self.tabWidget.addTab(self.speechToTextTab, 'Speech to Text')
        

    def actionSettingsClicked(self):
        dialog = SettingsDialog(self)
        dialog.exec()
            

    # Handles the event where the user closes the window with the X button.
    def closeEvent(self, event):
        self.odasLiveTab.closeEvent(event)
        self.speechToTextTab.closeEvent(event)
        event.accept()

    # Used by tab modules to tell the exception manager that an exception occured.    
    def emitToExceptionManager(self, exception):
        self.exceptionManager.signalException.emit(exception)
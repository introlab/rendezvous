import os
from pathlib import Path

from PyQt5.QtWidgets import QMainWindow
from src.app.gui.main_window_ui import Ui_MainWindow
from src.app.managers.exception_manager import ExceptionManager
from src.app.views.odas_live import OdasLive
from src.app.managers.settings_manager import SettingsManager
from src.app.views.transcription import Transcription
from src.app.dialogs.settings import Settings


class MainWindow(QMainWindow, Ui_MainWindow):

    rootDirectory = os.path.realpath(Path(__file__).parents[3])

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # Exception manager.
        self.exceptionManager = ExceptionManager()

        # Settings manager.
        self.settingsManager = SettingsManager()

        # Top options.
        self.actionSettings.triggered.connect(self.actionSettingsClicked)

        # Tabs of the main layout.
        self.odasLiveTab = OdasLive(parent=self)   
        self.tabWidget.addTab(self.odasLiveTab, 'Odas Live')

        self.transcriptionTab = Transcription(parent=self)   
        self.tabWidget.addTab(self.transcriptionTab, 'Transcription')
        

    def actionSettingsClicked(self):
        dialog = Settings(self)
        dialog.exec()
            

    # Handles the event where the user closes the window with the X button.
    def closeEvent(self, event):
        if event:
            self.odasLiveTab.closeEvent(event)
            self.transcriptionTab.closeEvent(event)
            event.accept()


    # Used by tab modules to tell the exception manager that an exception occured.    
    def emitToExceptionManager(self, exception):
        self.exceptionManager.signalException.emit(exception)
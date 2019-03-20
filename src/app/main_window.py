import os
from pathlib import Path

from PyQt5.QtWidgets import QMainWindow
from src.app.gui.main_window_ui import Ui_MainWindow

from src.app.managers.exceptions import Exceptions
from src.app.managers.settings import Settings

from src.app.views.conference import Conference
from src.app.views.transcription import Transcription
from src.app.views.change_settings import ChangeSettings


class MainWindow(QMainWindow, Ui_MainWindow):

    rootDirectory = str(Path(__file__).resolve().parents[3])

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # Exception manager.
        self.__exceptionsManager = Exceptions()

        # Settings manager.
        self.__settingsManager = Settings()

        # Tabs of the main layout.
        self.conferenceTab = Conference(parent=self)   
        self.tabWidget.addTab(self.conferenceTab, 'Conference')

        self.transcriptionTab = Transcription(parent=self)   
        self.tabWidget.addTab(self.transcriptionTab, 'Transcription')
        
        self.settingsTab = ChangeSettings(parent=self)   
        self.tabWidget.addTab(self.settingsTab, 'Settings')
  

    # Handles the event where the user closes the window with the X button.
    def closeEvent(self, event):
        if event:
            self.conferenceTab.closeEvent(event)
            self.transcriptionTab.closeEvent(event)
            event.accept()


    # Used by tab modules to tell the exception manager that an exception occured.    
    def emitToExceptionsManager(self, exception):
        self.__exceptionsManager.signalException.emit(exception)

    # Used by tab modules to set an application setting.
    def setSetting(self, setting, value):
        self.__settingsManager.setValue(setting, value)
    
    # Used by tab modules to get an application setting.
    def getSetting(self, setting):
        return self.__settingsManager.getValue(setting)

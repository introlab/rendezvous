import os
from pathlib import Path

from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from PyQt5.QtCore import pyqtSlot
from src.app.gui.main_window_ui import Ui_MainWindow

from src.app.managers.exceptions import Exceptions
from src.app.managers.settings import Settings

from src.app.components.side_bar import SideBar

from src.app.views.conference import Conference
from src.app.views.transcription import Transcription
from src.app.views.playback import Playback
from src.app.views.change_settings import ChangeSettings
from src.app.views.recording import Recording


class MainWindow(QMainWindow, Ui_MainWindow):

    rootDirectory = str(Path(__file__).resolve().parents[2])

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Exception manager.
        self.__exceptionsManager = Exceptions()

        # Settings manager.
        self.__settingsManager = Settings()

        # Initilization of the UI from QtCreator.
        self.setupUi(self)

        # Initialization of the application side bar.
        self.sideBar = SideBar() 
        self.sideBar.currentRowChanged.connect(self.onSideBarCurrentRowChanged) 
        self.mainLayout.addWidget(self.sideBar)

        # Initializaion of the views container.
        self.views = QStackedWidget()
        self.mainLayout.addWidget(self.views)
    
        # Views of the main layout.
        self.conferenceView = Conference(parent=self)
        self.sideBar.addItem('Conference')
        self.views.addWidget(self.conferenceView)
        self.sideBar.setCurrentRow(0)      

        self.recordingView = Recording(parent=self)
        self.sideBar.addItem('Recording')
        self.views.addWidget(self.recordingView) 

        self.transcriptionView = Transcription(parent=self)   
        self.sideBar.addItem('Transcription')
        self.views.addWidget(self.transcriptionView)

        self.playback = Playback(parent=self)   
        self.sideBar.addItem('Playback')
        self.views.addWidget(self.playback)

        self.settingsView = ChangeSettings(parent=self)   
        self.sideBar.addItem('Settings')
        self.views.addWidget(self.settingsView)


    # Handles the event where the user closes the window with the X button.
    def closeEvent(self, event):
        if event:
            self.conferenceView.closeEvent(event)
            self.transcriptionView.closeEvent(event)
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


    @pyqtSlot(int)
    def onSideBarCurrentRowChanged(self, i):
        self.views.setCurrentIndex(i)


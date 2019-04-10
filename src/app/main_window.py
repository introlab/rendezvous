import os
from pathlib import Path

from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from PyQt5.QtCore import pyqtSlot
from src.app.gui.main_window_ui import Ui_MainWindow

from src.app.components.side_bar import SideBar

from src.app.views.conference import Conference
from src.app.views.transcription import Transcription
from src.app.views.playback import Playback
from src.app.views.change_settings import ChangeSettings
from src.app.views.audio_processing import AudioProcessing
from src.app.views.recording import Recording

from src.app.services.odas.odas import Odas


class MainWindow(QMainWindow, Ui_MainWindow):

    rootDirectory = str(Path(__file__).resolve().parents[2])

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Initilization of the UI from QtCreator.
        self.setupUi(self)

        # Initialization of the application side bar.
        self.sideBar = SideBar() 
        self.sideBar.currentRowChanged.connect(self.onSideBarCurrentRowChanged) 
        self.mainLayout.addWidget(self.sideBar)

        # Initializaion of the views container.
        self.views = QStackedWidget()
        self.mainLayout.addWidget(self.views)
    
        odasServer = Odas(hostIP='127.0.0.1', portPositions=10020, portAudio=10030, isVerbose=False)
        odasServer.start()

        # Views of the main layout.
        self.conferenceView = Conference(odasServer, parent=self)
        self.sideBar.addItem('Conference')
        self.views.addWidget(self.conferenceView)
        self.sideBar.setCurrentRow(0)

        self.recordingView = Recording(odasServer, parent=self)
        self.sideBar.addItem('Recording')
        self.views.addWidget(self.recordingView) 

        self.audioProcessingView = AudioProcessing(parent=self)
        self.sideBar.addItem('Audio Processing')
        self.views.addWidget(self.audioProcessingView)

        self.transcriptionView = Transcription(parent=self)   
        self.sideBar.addItem('Transcription')
        self.views.addWidget(self.transcriptionView)

        self.playbackView = Playback(parent=self)   
        self.sideBar.addItem('Playback')
        self.views.addWidget(self.playbackView)

        self.settingsView = ChangeSettings(parent=self)   
        self.sideBar.addItem('Settings')
        self.views.addWidget(self.settingsView)


    # Handles the event where the user closes the window with the X button.
    def closeEvent(self, event):
        if event:
            self.conferenceView.closeEvent(event)
            self.recordingView.closeEvent(event)
            self.audioProcessingView.closeEvent(event)
            self.transcriptionView.closeEvent(event)
            self.playbackView.closeEvent(event)
            self.settingsView.closeEvent(event)
            event.accept()


    @pyqtSlot(int)
    def onSideBarCurrentRowChanged(self, i):
        self.views.setCurrentIndex(i)


from pathlib import Path

from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot, QTimer

from src.app.controllers.playback_controller import PlaybackController
from src.app.gui.playback_ui import Ui_Playback

class Playback(QWidget, Ui_Playback):

    rootDirectory = str(Path(__file__).resolve().parents[3])

    def __init__(self, parent=None):
        super(Playback, self).__init__(parent)
        self.setupUi(self)

        # Initilalization of the controller.
        self.playbackController = PlaybackController(parent=self)

        # Timer to update the UI.
        self.timer = QTimer(self)
        self.timer.setInterval(100)

        # Qt signal slots.
        self.timeSlider.sliderMoved.connect(self.setTime)
        self.timeSlider.sliderPressed.connect(self.setTime)
        self.volumeSlider.valueChanged.connect(self.setVolume)
        self.playPauseBtn.clicked.connect(self.onPlayPauseClicked)
        self.stopBtn.clicked.connect(self.onStopClicked)
        self.importMediaBtn.clicked.connect(self.onImportMediaClicked)
        self.playbackController.exception.connect(self.onException)
        self.timer.timeout.connect(self.updateUI)


    def closeEvent(self, event):
        if event:
            event.accept()


    def updateUI(self):
        mediaTime = int(self.playbackController.getTime())
        self.timeSlider.setValue(mediaTime)

        if not self.playbackController.isPlaying():
            self.timer.stop()

            if not self.playbackController.isPaused():
                self.playbackController.stop()


    def setTime(self):
        self.timer.stop()
        time = self.timeSlider.value()
        self.playbackController.setTime(time)
        self.timer.start()
    

    def setVolume(self, volume):
        self.playbackController.setVolume(volume)


    def onPlayPauseClicked(self):
        if self.playbackController.isPlaying():
            self.playbackController.pause()
            self.playPauseBtn.setText('Play')
            self.timer.stop()
        else:
            if self.mediaPlaying.text() == 'No Media Playing':
                return

            self.playbackController.play()
            self.playPauseBtn.setText('Pause')
            self.timer.start()

    
    def onStopClicked(self):
        self.playbackController.stop()
        self.playPauseBtn.setText('Play')


    @pyqtSlot()
    def onImportMediaClicked(self):
        try:
            mediaPath, _ = QFileDialog.getOpenFileName(parent=self, 
                                                        caption='Import Media File', 
                                                        directory=self.window().rootDirectory,
                                                        options=QFileDialog.DontUseNativeDialog)
            if mediaPath:
                self.playbackController.loadMediaFile(mediaPath, self.videoFrame.winId())
                self.mediaPlaying.setText(self.playbackController.getPlayingMediaName())

        except Exception as e:
            self.window().emitToExceptionsManager(e)
            self.mediaPath.setText('No Media Playing')


    @pyqtSlot(Exception)
    def onException(self, e):
        self.window().emitToExceptionsManager(e)


from pathlib import Path

from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot, QTimer

from src.app.controllers.playback_controller import PlaybackController
from src.app.gui.playback_ui import Ui_Playback

class Playback(QWidget, Ui_Playback):

    rootDirectory = str(Path(__file__).resolve().parents[3])
    __positionResolution = 1000


    def __init__(self, parent=None):
        super(Playback, self).__init__(parent)
        self.setupUi(self)

        # Initilalization of the controller.
        self.playbackController = PlaybackController(parent=self)

        # Timer to update the UI.
        self.uiUpdateTimer = QTimer(self)
        self.uiUpdateTimer.setInterval(100)


        # Populate UI.
        self.volumeSlider.setMaximum(100)
        self.volumeSlider.setValue(50)
        self.volumeSlider.setToolTip('Audio Slider')
        self.volumeSlider.sliderMoved.connect(self.onVolume)
        self.mediaPositionSlider.setMaximum(self.__positionResolution)


        # Qt signal slots.
        self.mediaPositionSlider.sliderMoved.connect(self.onMediaPositionMoved)
        self.mediaPositionSlider.sliderPressed.connect(self.onMediaPositionPressed)
        self.mediaPositionSlider.sliderReleased.connect(self.onMediaPositionReleased)
        self.playPauseBtn.clicked.connect(self.onPlayPauseClicked)
        self.stopBtn.clicked.connect(self.onStopClicked)
        self.importMediaBtn.clicked.connect(self.onImportMediaClicked)
        self.uiUpdateTimer.timeout.connect(self.onUiUpdateTimerTimeout)
        self.playbackController.mediaPlayerEndReached.connect(self.onMediaPlayerEndReached)

    def closeEvent(self, event):
        if event:
            event.accept()


    def emitException(self, e):
        self.window().emitToExceptionsManager(e)


    def updatePlayingMediaName(self, name):
        self.mediaPlaying.setText(name if name != None else 'No Media Playing')


    @pyqtSlot()
    def onUiUpdateTimerTimeout(self):
        if not self.playbackController.isPlaying():
            self.uiUpdateTimer.stop()

            if not self.playbackController.isPaused():
                self.playbackController.stop()
                self.playPauseBtn.setText('Play')

         # Update the value of the position slider.
        mediaPosition = self.playbackController.getPosition()
        mediaPosition = int(mediaPosition * self.__positionResolution) if mediaPosition != self.playbackController.errorCode else 0 
        self.mediaPositionSlider.setValue(mediaPosition) 


    @pyqtSlot(int)
    def onMediaPositionMoved(self, value):
        self.playbackController.setPosition(self.mediaPositionSlider.value() / float(self.__positionResolution))


    @pyqtSlot()
    def onMediaPositionPressed(self):
        self.uiUpdateTimer.stop()


    @pyqtSlot()
    def onMediaPositionReleased(self):
        self.uiUpdateTimer.start()
    

    @pyqtSlot(int)
    def onVolume(self, volume):
        self.playbackController.setVolume(volume)


    @pyqtSlot()
    def onPlayPauseClicked(self):
        try:
            if self.playbackController.isPlaying():
                self.uiUpdateTimer.stop()
                self.playbackController.pause()
                self.playPauseBtn.setText('Play')
            else:
                self.uiUpdateTimer.start()
                self.playbackController.play()
                self.playPauseBtn.setText('Pause')

        except Exception as e:
            self.emitException(e)
    
    
    @pyqtSlot()
    def onStopClicked(self):
        self.playbackController.stop()
        self.playPauseBtn.setText('Play')


    @pyqtSlot()
    def onMediaPlayerEndReached(self):
        self.mediaPositionSlider.setValue(self.mediaPositionSlider.maximum())


    @pyqtSlot()
    def onImportMediaClicked(self):
        try:
            mediaPath, _ = QFileDialog.getOpenFileName(parent=self, 
                                                       caption='Import Media File', 
                                                       directory=self.window().rootDirectory,
                                                       options=QFileDialog.DontUseNativeDialog)
            if mediaPath:
                self.playbackController.loadMediaFile(mediaPath, self.videoFrame.winId())
                self.playbackController.setVolume(self.volumeSlider.value())
                self.updatePlayingMediaName(self.playbackController.getPlayingMediaName())
        except Exception as e:
            self.emitException(e)
            self.updatePlayingMediaName('No Media Playing')


import platform
import os

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

import vlc


class PlaybackController(QObject):  

    exception = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(PlaybackController, self).__init__(parent)

        self.__instance = vlc.Instance('-q --no-xlib')

        self.__media = None

        self.__mediaPlayer = self.__instance.media_player_new()

        self.__isPaused = False


    def play(self):
        try:
            self.__mediaPlayer.play()
            self.__isPaused = False
        
        except Exception as e:
            self.onException(e)


    def isPlaying(self):
        return self.__mediaPlayer.is_playing()


    def pause(self):
        try:
            self.__mediaPlayer.pause()
            self.__isPaused = True

        except Exception as e:
            self.onException(e)


    def isPaused(self):
        return self.__isPaused


    def stop(self):
        try:
            self.__mediaPlayer.stop()

        except Exception as e:
            self.onException(e)


    def setVolume(self, volume):
        try:
            self.__mediaPlayer.audio_set_volume(volume)

        except Exception as e:
            self.onException(e)

    def setTime(self, time):
        try:
            self.__mediaPlayer.set_position(time / 1000.0)

        except Exception as e:
            self.onException(e)            


    def getTime(self):
        return self.__mediaPlayer.get_position() * 1000



    def getPlayingMediaName(self):
        return self.__media.get_meta(0)

    def loadMediaFile(self, file, winId):
        try:
            if not os.path.exists(file):
                raise Exception('No media file found at : {}'.format(file))

            self.__media = self.__instance.media_new(file)
            self.__mediaPlayer.set_media(self.__media)  
            self.__media.parse()

            if platform.system() == "Linux": # for Linux using the X Server
                self.__mediaPlayer.set_xwindow(int(winId))
            elif platform.system() == "Windows": # for Windows
                self.__mediaPlayer.set_hwnd(int(winId))
            elif platform.system() == "Darwin": # for MacOS
                self.__mediaPlayer.set_nsobject(int(winId))

        except Exception as e:
            self.onException(e)

    def onException(self, e):
        self.exception.emit(e)


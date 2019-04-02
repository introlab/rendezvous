import platform
import os

from PyQt5.QtCore import QObject, pyqtSignal

import vlc


class PlaybackController(QObject):  

    mediaPlayerEndReached = pyqtSignal()
    errorCode = -1
    __isPaused = False
    __media = None

    def __init__(self, parent=None):
        super(PlaybackController, self).__init__(parent)

        self.__instance = vlc.Instance('-q --no-xlib')
        self.__mediaPlayer = self.__instance.media_player_new()

        # self.events = self.__mediaPlayer.event_manager()
        self.__mediaPlayer.event_manager().event_attach(vlc.EventType.MediaPlayerEndReached, self.onEventManage)


    def play(self):
        if self.__mediaPlayer.play() == self.errorCode:
            raise Exception('Failed to play media {}'.format(self.__media))
        else:
            self.__isPaused = False


    def isPlaying(self):
        return self.__mediaPlayer.is_playing()


    def pause(self):
        if self.isPlaying():
            self.__mediaPlayer.pause()
            self.__isPaused = True


    def isPaused(self):
        return self.__isPaused


    def stop(self):
        self.__mediaPlayer.stop()


    def setVolume(self, volume):
        if self.__mediaPlayer.get_media() != None:
            self.__mediaPlayer.audio_set_volume(volume)


    def getVolume(self):
        return self.__mediaPlayer.audio_get_volume()
 

    def setPosition(self, position):
        if self.__mediaPlayer.get_media() != None:
            self.__mediaPlayer.set_position(position)       


    def getPosition(self):
        return self.__mediaPlayer.get_position()    


    def getPlayingMediaName(self):
        return self.__media.get_meta(0)


    def loadMediaFile(self, file, winId):
        if not os.path.exists(file):
            raise Exception('No media file found at : {}'.format(file))

        self.__media = self.__instance.media_new(file)
        if self.__media == None:
            raise Exception('Media file {} can\'t be loaded'.format(file))
        self.__media.parse()

        self.__mediaPlayer.set_media(self.__media)  

        if platform.system() == "Linux":
            self.__mediaPlayer.set_xwindow(int(winId))
        else:
            raise Exception('Playback is not supported on this operating system')


    def onEventManage(self, event):
        if event.type == vlc.EventType.MediaPlayerEndReached:
            self.mediaPlayerEndReached.emit()


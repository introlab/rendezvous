import platform
import os

from PyQt5.QtCore import QObject, pyqtSignal

import vlc


class PlaybackController(QObject):  
    
    mediaPlayerPlayed = pyqtSignal()
    mediaPlayerEndReached = pyqtSignal()
    errorCode = -1
    __isPaused = False
    __enableSubtitle = True
    __media = None

    def __init__(self, parent=None, winId=0):
        super(PlaybackController, self).__init__(parent)

        self.__instance = vlc.Instance('-q --no-xlib')
        self.__mediaPlayer = self.__instance.media_player_new()

        if platform.system() == "Linux":
            self.__mediaPlayer.set_xwindow(winId)
        else:
            raise Exception('Playback is not supported on this operating system')   

        self.events = self.__mediaPlayer.event_manager()
        self.events.event_attach(vlc.EventType.MediaPlayerEndReached, self.onEventManage)
        self.events.event_attach(vlc.EventType.MediaPlayerEncounteredError, self.onEventManage)
        self.events.event_attach(vlc.EventType.MediaPlayerPlaying, self.onEventManage)


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


    def toggleSubtitleState(self):
        self.__enableSubtitle = not self.__enableSubtitle
        self.updateEnableSubtitle()


    def updateEnableSubtitle(self):    
        if self.__enableSubtitle:
            subtitlesCount = self.__mediaPlayer.video_get_spu_count()
            if subtitlesCount != -1:
                self.__mediaPlayer.video_set_spu(subtitlesCount)
        else:
            self.__mediaPlayer.video_set_spu(-1)
        print("ASDFAD")


    def getSubtitleState(self):
        # Qt::CheckState is 2 when it's checked, else 0.
        return 2 if self.__enableSubtitle else 0

        
    def loadMediaFile(self, file):
        if not os.path.exists(file):
            raise Exception('No media file found at : {}'.format(file))

        self.__media = self.__instance.media_new(file)
        if self.__media == None:
            raise Exception('Media file {} can\'t be loaded'.format(file))

        # This function could block indefinitely. Use libvlc_media_parse_with_options() instead
        # We use parse(), since it's the only option on the version of the lib supported by the Jetson Tx2.
        self.__media.parse()
        self.__mediaPlayer.set_media(self.__media)


    def onEventManage(self, event):
        if event.type == vlc.EventType.MediaPlayerEndReached:
            self.mediaPlayerEndReached.emit()
        elif event.type == vlc.EventType.MediaPlayerPlaying:
            self.mediaPlayerPlayed.emit()
            self.updateEnableSubtitle()
        elif event.type == vlc.EventType.MediaPlayerEncounteredError: 
            raise Exception(event)


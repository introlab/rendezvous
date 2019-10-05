#ifndef I_VIDEO_PLAYER_H
#define I_VIDEO_PLAYER_H

#include <QMediaPlayer>

class QVideoWidget;

namespace Model
{

class IVideoPlayer
{
    public:
        virtual ~IVideoPlayer() {}
        virtual void openFile() = 0;
        virtual void play() = 0;
        virtual void setPosition(int position) = 0;
        virtual void setVideoOutput(QVideoWidget *videoOutput);

     signals:
        void stateChanged(QMediaPlayer::State state);
        void positionChanged(qint64 position);
        void durationChanged(qint64 duration);
        void errorOccured();
};

} // Model

#endif // I_VIDEO_PLAYER_H

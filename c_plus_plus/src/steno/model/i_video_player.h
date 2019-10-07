#ifndef I_VIDEO_PLAYER_H
#define I_VIDEO_PLAYER_H

#include <QMediaPlayer>
#include <QWidget>

class QVideoWidget;

namespace Model
{

class IVideoPlayer : public QWidget
{
    Q_OBJECT

    public:
        IVideoPlayer(QWidget *parent = nullptr) : QWidget(parent) {}
        virtual ~IVideoPlayer() {}
        virtual void openFile() = 0;
        virtual void play() = 0;
        virtual void setPosition(int position) = 0;
        virtual void setVideoOutput(QVideoWidget *videoOutput) = 0;
        virtual void setVolume(int volume) = 0;
        virtual int volume() const = 0;

     signals:
        void stateChanged(QMediaPlayer::State state);
        void positionChanged(qint64 position);
        void durationChanged(qint64 duration);
        void volumeChanged(int volume);
        void errorOccured(QString error);
        void setUrlCompleted();
};

} // Model

#endif // I_VIDEO_PLAYER_H

#ifndef I_MEDIA_PLAYER_H
#define I_MEDIA_PLAYER_H

#include <QMediaPlayer>
#include <QWidget>

class QVideoWidget;

namespace Model
{

class IMediaPlayer : public QWidget
{
Q_OBJECT

public:
    IMediaPlayer(QWidget *parent = nullptr) : QWidget(parent) {}
    virtual void setMedia(const QUrl &url) = 0;
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
    void errorOccured(const QString& error);
};

} // Model

#endif // I_MEDIA_PLAYER_H

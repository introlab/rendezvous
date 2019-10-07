#include "video_player.h"

#include <QVideoWidget>

namespace Model
{

VideoPlayer::VideoPlayer(QWidget *parent)
    : IVideoPlayer(parent)
    , m_mediaPlayer(new QMediaPlayer(this, QMediaPlayer::VideoSurface))

{
    connect(m_mediaPlayer, &QMediaPlayer::stateChanged, [=](QMediaPlayer::State state){ emit stateChanged(state); });
    connect(m_mediaPlayer, &QMediaPlayer::positionChanged, [=](qint64 position){ emit positionChanged(position); });
    connect(m_mediaPlayer, &QMediaPlayer::durationChanged, [=](qint64 duration){ emit durationChanged(duration); });
    connect(m_mediaPlayer, &QMediaPlayer::volumeChanged, [=](int volume){ emit volumeChanged(volume); });
    connect(m_mediaPlayer, QOverload<QMediaPlayer::Error>::of(&QMediaPlayer::error), [=]{ onErrorOccured(); });
}

VideoPlayer::~VideoPlayer()
{
    delete m_mediaPlayer;
}

void VideoPlayer::setVideoOutput(QVideoWidget *videoOutput)
{
    m_mediaPlayer->setVideoOutput(videoOutput);
}

void VideoPlayer::setVolume(int volume)
{
    m_mediaPlayer->setVolume(volume);
}

int VideoPlayer::volume() const
{
    return m_mediaPlayer->volume();
}

void VideoPlayer::setMedia(const QUrl &url)
{
    m_mediaPlayer->setMedia(url);
}

void VideoPlayer::play()
{
    switch (m_mediaPlayer->state())
    {
        case QMediaPlayer::PlayingState:
            m_mediaPlayer->pause();
            break;
        default:
            m_mediaPlayer->play();
            break;
    }
}

void VideoPlayer::setPosition(int position)
{
    m_mediaPlayer->setPosition(position);
}

void VideoPlayer::onErrorOccured()
{
    const QString errorString = m_mediaPlayer->errorString();
    QString message = "Error: ";

    if (errorString.isEmpty())
    {
        message += " #" + QString::number(int(m_mediaPlayer->error()));
    }
    else
    {
        message += errorString;
    }

    emit errorOccured(message);
}

} // Model

#include "media_player.h"

#include <QFileInfo>
#include <QUrl>
#include <QVideoWidget>

namespace Model
{
MediaPlayer::MediaPlayer(QWidget *parent)
    : IMediaPlayer(parent)
    , m_mediaPlayer(new QMediaPlayer(this, QMediaPlayer::VideoSurface))

{
    connect(&m_mediaPlayer, &QMediaPlayer::stateChanged, [=](QMediaPlayer::State state) { onStateChanged(state); });
    connect(&m_mediaPlayer, &QMediaPlayer::positionChanged, [=](qint64 position) { emit positionChanged(position); });
    connect(&m_mediaPlayer, &QMediaPlayer::durationChanged, [=](qint64 duration) { emit durationChanged(duration); });
    connect(&m_mediaPlayer, &QMediaPlayer::volumeChanged, [=](int volume) { emit volumeChanged(volume); });
    connect(&m_mediaPlayer, QOverload<QMediaPlayer::Error>::of(&QMediaPlayer::error), [=] { onErrorOccured(); });
    connect(&m_subtitles, &Subtitles::subtitleChanged,
            [=](const QString &subtitle) { emit subtitleChanged(subtitle); });
}

void MediaPlayer::setVideoOutput(QVideoWidget *videoOutput)
{
    m_mediaPlayer.setVideoOutput(videoOutput);
}

void MediaPlayer::setVolume(int volume)
{
    m_mediaPlayer.setVolume(volume);
}

int MediaPlayer::volume() const
{
    return m_mediaPlayer.volume();
}

/**
 * @brief Set the current media file to play.
 * @param [IN] url - Path to the file to play.
 */
void MediaPlayer::setMedia(const QUrl &url)
{
    m_mediaPlayer.setMedia(url);

    QFileInfo fileInfo(url.toLocalFile());
    QString srtFilePath = fileInfo.path() + "/" + fileInfo.completeBaseName() + ".srt";

    if (QFileInfo::exists(srtFilePath))
    {
        m_subtitles.open(srtFilePath);
    }

    // We do that to load the first frame of the video when the user open a media file.
    m_mediaPlayer.play();
    m_mediaPlayer.pause();
    setPosition(0);
}

void MediaPlayer::play()
{
    switch (m_mediaPlayer.state())
    {
        case QMediaPlayer::PlayingState:
            m_mediaPlayer.pause();
            m_subtitles.pause();
            break;
        default:
            m_mediaPlayer.play();
            m_subtitles.play();
            break;
    }
}

/**
 * @brief Set the current position in time of the media file.
 * @param [IN] position
 */
void MediaPlayer::setPosition(int position)
{
    m_mediaPlayer.setPosition(position);
    m_subtitles.setCurrentTime(position);
}

/**
 * @brief what to do when the QMediaPlayer state changed
 * @param [IN] state
 */
void MediaPlayer::onStateChanged(QMediaPlayer::State state)
{
    if (state == QMediaPlayer::State::StoppedState)
    {
        m_subtitles.stop();
        emit positionChanged(0);
    }
    emit stateChanged(state);
}

void MediaPlayer::onErrorOccured()
{
    const QString errorString = m_mediaPlayer.errorString();
    QString message = "Error: ";

    if (errorString.isEmpty())
    {
        message += " #" + QString::number(int(m_mediaPlayer.error()));
    }
    else
    {
        message += errorString;
    }

    emit errorOccured(message);
}

}    // namespace Model

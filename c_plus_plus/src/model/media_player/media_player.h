#ifndef MEDIA_PLAYER_H
#define MEDIA_PLAYER_H

#include "i_media_player.h"
#include "subtitles/subtitles.h"

#include <QMediaPlayer>
#include <QUrl>
#include <QWidget>

class QUrl;

namespace Model
{

class MediaPlayer : public IMediaPlayer
{
public:
    explicit MediaPlayer(QWidget *parent = nullptr);
    void setMedia(const QUrl &url) override;
    void play() override;
    void setPosition(int position) override;
    void setVideoOutput(QVideoWidget *videoOutput) override;
    void setVolume(int volume) override;
    int volume() const override;

private slots:
    void onStateChanged(QMediaPlayer::State state);
    void onErrorOccured();

private:
    QMediaPlayer m_mediaPlayer;
    Subtitles m_subtitles;
};

} // Model

#endif // MEDIA_PLAYER_H

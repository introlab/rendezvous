#ifndef MEDIA_PLAYER_H
#define MEDIA_PLAYER_H

#include "model/i_media_player.h"

#include <QMediaPlayer>
#include <QWidget>

class QUrl;

namespace Model
{

class MediaPlayer : public IMediaPlayer
{
    public:
        explicit MediaPlayer(QWidget *parent = nullptr);
        virtual ~MediaPlayer() override;
        virtual void setMedia(const QUrl &url) override;
        virtual void play() override;
        virtual void setPosition(int position) override;
        virtual void setVideoOutput(QVideoWidget *videoOutput) override;
        virtual void setVolume(int volume) override;
        virtual int volume() const override;

    private:
        void onErrorOccured();

        QMediaPlayer* m_mediaPlayer;
};

} // Model

#endif // MEDIA_PLAYER_H

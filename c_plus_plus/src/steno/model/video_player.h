#ifndef VIDEO_PLAYER_H
#define VIDEO_PLAYER_H

#include "model/i_video_player.h"

#include <QMediaPlayer>
#include <QWidget>

class QUrl;

namespace Model
{

class VideoPlayer : public IVideoPlayer
{
    public:
        explicit VideoPlayer(QWidget *parent = nullptr);
        virtual ~VideoPlayer() override;
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

#endif // VIDEO_PLAYER_H

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
        virtual ~VideoPlayer();
        virtual void openFile();
        virtual void play();
        virtual void setPosition(int position);
        virtual void setVideoOutput(QVideoWidget *videoOutput);

    private:
        void onErrorOccured();
        void setUrl(const QUrl &url);

        QMediaPlayer* m_mediaPlayer;
};

} // Model

#endif // VIDEO_PLAYER_H

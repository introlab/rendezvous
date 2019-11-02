#ifndef MEDIA_PLAYER_VIEW_H
#define MEDIA_PLAYER_VIEW_H

#include "model/media_player/i_media_player.h"
#include "view/views/abstract_view.h"

#include <QMediaPlayer>

namespace Ui
{
class MediaPlayerView;
}

namespace View
{
class MediaPlayerView : public AbstractView
{
   public:
    explicit MediaPlayerView(Model::IMediaPlayer &videoPlayer, QWidget *parent = nullptr);

   private slots:
    void onMediaStateChanged(QMediaPlayer::State state);
    void onPositionChanged(qint64 position);
    void onDurationChanged(qint64 duration);
    void onSubtitleChanged(const QString &subtitle);
    void onErrorOccured(const QString &error);

   private:
    void openFile();
    void setVolume(int);
    int volume() const;

    Ui::MediaPlayerView *m_ui;
    Model::IMediaPlayer &m_videoPlayer;

    const uint8_t m_maxVolume = 100;
};

}    // namespace View

#endif    // MEDIA_PLAYER_VIEW_H

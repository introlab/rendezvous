#ifndef MEDIA_PLAYER_VIEW_H
#define MEDIA_PLAYER_VIEW_H

#include "model/media_player/i_media_player.h"
#include "view/views/abstract_view.h"

#include <memory>

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
    explicit MediaPlayerView(std::shared_ptr<Model::IMediaPlayer> mediaPlayer, QWidget *parent = nullptr);

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
    std::shared_ptr<Model::IMediaPlayer> m_mediaPlayer;

    const uint8_t m_maxVolume = 100;
};

}    // namespace View

#endif    // MEDIA_PLAYER_VIEW_H

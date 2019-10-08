#ifndef MEDIA_PLAYER_VIEW_H
#define MEDIA_PLAYER_VIEW_H

#include <QMediaPlayer>

#include "view/views/abstract_view.h"

namespace Ui { class MediaPlayerView; }
namespace Model { class IMediaPlayer; }

namespace View
{

class MediaPlayerView : public AbstractView
{
public:
    explicit MediaPlayerView(Model::IMediaPlayer& videoPlayer, QWidget *parent = nullptr);

public slots:
    void onMediaStateChanged(QMediaPlayer::State state);
    void onPositionChanged(qint64 position);
    void onDurationChanged(qint64 duration);
    void onErrorOccured(const QString &error);

private:
    void openFile();
    void setVolume(int);
    int volume() const;

    Ui::MediaPlayerView *m_ui;
    Model::IMediaPlayer &m_videoPlayer;

    const uint8_t m_maxVolume = 100;
};

} // View

#endif // MEDIA_PLAYER_VIEW_H

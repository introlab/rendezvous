#ifndef PLAYBACK_VIEW_H
#define PLAYBACK_VIEW_H

#include <QMediaPlayer>

#include "view/views/abstract_view.h"

namespace Ui { class PlaybackView; }
namespace Model { class IVideoPlayer; }

namespace View
{

class PlaybackView : public AbstractView
{
    public:
        explicit PlaybackView(Model::IVideoPlayer& videoPlayer, QWidget *parent = nullptr);
        virtual ~PlaybackView();

    public slots:
        void onMediaStateChanged(QMediaPlayer::State state);
        void onPositionChanged(qint64 position);
        void onDurationChanged(qint64 duration);
        void onErrorOccured(QString error);
        void onOpenFileAccepted();

    private:
        int volume() const;
        void setVolume(int);

        Ui::PlaybackView *m_ui;
        Model::IVideoPlayer &m_videoPlayer;
};

} // View

#endif // PLAYBACK_VIEW_H

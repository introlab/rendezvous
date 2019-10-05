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
        explicit PlaybackView(QWidget *parent = nullptr);
        virtual ~PlaybackView();

        void setUrl(const QUrl &url);
        void openFile();
        void play();

    private:
        void mediaStateChanged(QMediaPlayer::State state);
        void positionChanged(qint64 position);
        void durationChanged(qint64 duration);
        void setPosition(int position);
        void handleError();

        Ui::PlaybackView *m_ui;

        QMediaPlayer* m_mediaPlayer;
};

} // View

#endif // PLAYBACK_VIEW_H

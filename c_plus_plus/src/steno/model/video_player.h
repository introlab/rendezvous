#ifndef VIDEO_PLAYER_H
#define VIDEO_PLAYER_H

#include <QMediaPlayer>
#include <QWidget>

class QUrl;

namespace Model
{

class VideoPlayer : public QWidget
{
    Q_OBJECT

    public:
        explicit VideoPlayer(QWidget *parent = nullptr);
        virtual ~VideoPlayer();

        void setUrl(const QUrl &url);
        void openFile();
        void play();

    private:
        void mediaStateChanged(QMediaPlayer::State state);
        void positionChanged(qint64 position);
        void durationChanged(qint64 duration);
        void setPosition(int position);
        void handleError();

        QMediaPlayer* m_mediaPlayer;

        //void errorOccured(QString error);
};

} // Model

#endif // VIDEO_PLAYER_H

#include "video_player.h"

#include <QtWidgets>
#include <QVideoWidget>

namespace Model
{

VideoPlayer::VideoPlayer(QWidget *parent)
    : QWidget(parent)
    , m_mediaPlayer(new QMediaPlayer(this, QMediaPlayer::VideoSurface))

{
}

VideoPlayer::~VideoPlayer()
{

}

} // Model

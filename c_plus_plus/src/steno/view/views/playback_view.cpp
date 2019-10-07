#include "playback_view.h"
#include "ui_playback_view.h"

#include "model/i_video_player.h"

#include <QtWidgets>

namespace View
{

PlaybackView::PlaybackView(Model::IVideoPlayer &videoPlayer, QWidget *parent)
    : AbstractView("Playback", parent)
    , m_ui(new Ui::PlaybackView)
    , m_videoPlayer(videoPlayer)
{
    m_ui->setupUi(this);
    m_ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));

    setVolume(m_videoPlayer.volume());

    m_videoPlayer.setVideoOutput(m_ui->videoWidget);

    connect(m_ui->openButton, &QAbstractButton::clicked, [=]{ m_videoPlayer.openFile(); });
    connect(m_ui->playButton, &QAbstractButton::clicked, [=]{ m_videoPlayer.play(); });
    connect(m_ui->positionSlider, &QAbstractSlider::sliderMoved, [=](int position){ m_videoPlayer.setPosition(position); });
    connect(m_ui->volumeSlider, &QSlider::valueChanged, [=]{ m_videoPlayer.setVolume(volume()); });
    
    connect(&m_videoPlayer, &Model::IVideoPlayer::stateChanged, [=](QMediaPlayer::State state){ onMediaStateChanged(state); });
    connect(&m_videoPlayer, &Model::IVideoPlayer::positionChanged, [=](qint64 position){ onPositionChanged(position); });
    connect(&m_videoPlayer, &Model::IVideoPlayer::durationChanged, [=](qint64 duration){ onDurationChanged(duration); });
    connect(&m_videoPlayer, &Model::IVideoPlayer::volumeChanged, [=](int volume){ setVolume(volume); });
    connect(&m_videoPlayer, &Model::IVideoPlayer::errorOccured, [=](QString error){ onErrorOccured(error); });
    connect(&m_videoPlayer, &Model::IVideoPlayer::setUrlCompleted, [=]{ onOpenFileAccepted(); });
}

PlaybackView::~PlaybackView()
{
    delete m_ui;
}

void PlaybackView::onMediaStateChanged(QMediaPlayer::State state)
{
    switch(state)
    {
        case QMediaPlayer::PlayingState:
            m_ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
            break;
        default:
            m_ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
            break;
    }
}

void PlaybackView::onPositionChanged(qint64 position)
{
    m_ui->positionSlider->setValue(static_cast<int>(position));
}

void PlaybackView::onDurationChanged(qint64 duration)
{
    m_ui->positionSlider->setRange(0, static_cast<int>(duration));
}

void PlaybackView::onErrorOccured(QString error)
{
    m_ui->playButton->setEnabled(false);
    m_ui->errorLabel->setText(error);
}

void PlaybackView::onOpenFileAccepted()
{
    m_ui->errorLabel->setText(QString());
    m_ui->playButton->setEnabled(true);
}

int PlaybackView::volume() const
{
    qreal linearVolume = QAudio::convertVolume(m_ui->volumeSlider->value() / qreal(100),
                                               QAudio::LogarithmicVolumeScale,
                                               QAudio::LinearVolumeScale);

    return qRound(linearVolume * 100);
}

void PlaybackView::setVolume(int volume)
{
    qreal logarithmicVolume = QAudio::convertVolume(volume / qreal(100),
                                                    QAudio::LinearVolumeScale,
                                                    QAudio::LogarithmicVolumeScale);

    m_ui->volumeSlider->setValue(qRound(logarithmicVolume * 100));
}

} // View

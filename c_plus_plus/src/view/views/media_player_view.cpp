#include "media_player_view.h"
#include "ui_media_player_view.h"

#include "model/media_player/i_media_player.h"

#include <QFileDialog>
#include <QStandardPaths>
#include <QStyle>

namespace View
{
MediaPlayerView::MediaPlayerView(Model::IMediaPlayer& videoPlayer, QWidget* parent)
    : AbstractView("Media Player", parent), m_ui(new Ui::MediaPlayerView), m_videoPlayer(videoPlayer)
{
    m_ui->setupUi(this);
    m_ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));

    setVolume(m_videoPlayer.volume());

    m_videoPlayer.setVideoOutput(m_ui->videoWidget);

    connect(m_ui->openButton, &QAbstractButton::clicked, [=] { openFile(); });
    connect(m_ui->playButton, &QAbstractButton::clicked, [=] { m_videoPlayer.play(); });
    connect(m_ui->positionSlider, &QAbstractSlider::sliderMoved,
            [=](int position) { m_videoPlayer.setPosition(position); });
    connect(m_ui->volumeSlider, &QSlider::valueChanged, [=] { m_videoPlayer.setVolume(volume()); });

    connect(&m_videoPlayer, &Model::IMediaPlayer::stateChanged,
            [=](QMediaPlayer::State state) { onMediaStateChanged(state); });
    connect(&m_videoPlayer, &Model::IMediaPlayer::positionChanged,
            [=](qint64 position) { onPositionChanged(position); });
    connect(&m_videoPlayer, &Model::IMediaPlayer::durationChanged,
            [=](qint64 duration) { onDurationChanged(duration); });
    connect(&m_videoPlayer, &Model::IMediaPlayer::volumeChanged, [=](int volume) { setVolume(volume); });
    connect(&m_videoPlayer, &Model::IMediaPlayer::errorOccured, [=](const QString& error) { onErrorOccured(error); });
}

void MediaPlayerView::onMediaStateChanged(QMediaPlayer::State state)
{
    switch (state)
    {
        case QMediaPlayer::PlayingState:
            m_ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
            break;
        default:
            m_ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
            break;
    }
}

void MediaPlayerView::onPositionChanged(qint64 position) { m_ui->positionSlider->setValue(static_cast<int>(position)); }
void MediaPlayerView::onDurationChanged(qint64 duration)
{
    m_ui->positionSlider->setRange(0, static_cast<int>(duration));
}

void MediaPlayerView::onErrorOccured(const QString& error)
{
    m_ui->playButton->setEnabled(false);
    m_ui->errorLabel->setText(error);
}

void MediaPlayerView::openFile()
{
    QFileDialog fileDialog(this);
    fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
    fileDialog.setWindowTitle(tr("Open Media File"));

    fileDialog.setDirectory(
        QStandardPaths::standardLocations(QStandardPaths::MoviesLocation).value(0, QDir::homePath()));
    if (fileDialog.exec() == QDialog::Accepted)
    {
        m_videoPlayer.setMedia(fileDialog.selectedUrls().constFirst());
        m_ui->errorLabel->setText(QString());
        m_ui->playButton->setEnabled(true);
    }
}

int MediaPlayerView::volume() const
{
    qreal linearVolume = QAudio::convertVolume(m_ui->volumeSlider->value() / qreal(m_maxVolume),
                                               QAudio::LogarithmicVolumeScale, QAudio::LinearVolumeScale);

    return qRound(linearVolume * m_maxVolume);
}

void MediaPlayerView::setVolume(int volume)
{
    qreal logarithmicVolume =
        QAudio::convertVolume(volume / qreal(m_maxVolume), QAudio::LinearVolumeScale, QAudio::LogarithmicVolumeScale);

    m_ui->volumeSlider->setValue(qRound(logarithmicVolume * m_maxVolume));
}

}    // View

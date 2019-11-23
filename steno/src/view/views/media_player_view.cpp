#include "media_player_view.h"
#include "ui_media_player_view.h"

#include "model/media_player/i_media_player.h"

#include <QFileDialog>
#include <QStandardPaths>
#include <QStyle>

namespace View
{
MediaPlayerView::MediaPlayerView(std::shared_ptr<Model::IMediaPlayer> mediaPlayer, QWidget* parent)
    : AbstractView("Media Player", parent)
    , m_ui(new Ui::MediaPlayerView)
    , m_mediaPlayer(mediaPlayer)
{
    m_ui->setupUi(this);
    m_ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));

    setVolume(m_mediaPlayer->volume());

    m_mediaPlayer->setVideoOutput(m_ui->videoWidget);

    connect(m_ui->openButton, &QAbstractButton::clicked, [=] { openFile(); });
    connect(m_ui->playButton, &QAbstractButton::clicked, [=] { m_mediaPlayer->play(); });
    connect(m_ui->positionSlider, &QAbstractSlider::sliderMoved,
            [=](int position) { m_mediaPlayer->setPosition(position); });
    connect(m_ui->volumeSlider, &QSlider::valueChanged, [=] { m_mediaPlayer->setVolume(volume()); });

    connect(m_mediaPlayer.get(), &Model::IMediaPlayer::stateChanged,
            [=](QMediaPlayer::State state) { onMediaStateChanged(state); });
    connect(m_mediaPlayer.get(), &Model::IMediaPlayer::positionChanged,
            [=](qint64 position) { onPositionChanged(position); });
    connect(m_mediaPlayer.get(), &Model::IMediaPlayer::durationChanged,
            [=](qint64 duration) { onDurationChanged(duration); });
    connect(m_mediaPlayer.get(), &Model::IMediaPlayer::volumeChanged, [=](int volume) { setVolume(volume); });
    connect(m_mediaPlayer.get(), &Model::IMediaPlayer::subtitleChanged,
            [=](QString subtitle) { onSubtitleChanged(subtitle); });
    connect(m_mediaPlayer.get(), &Model::IMediaPlayer::errorOccured,
            [=](const QString& error) { onErrorOccured(error); });
}

/**
 * @brief Callback when the QMediaPlayer current state changed.
 * Currently change the playButton icon.
 * @param [IN] state
 */
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

/**
 * @brief Change the slider position representing the media file progression in time.
 * @param position
 */
void MediaPlayerView::onPositionChanged(qint64 position)
{
    m_ui->positionSlider->setValue(static_cast<int>(position));
}

/**
 * @brief Change the media duration in the UI.
 * @param duration
 */
void MediaPlayerView::onDurationChanged(qint64 duration)
{
    m_ui->positionSlider->setRange(0, static_cast<int>(duration));
}

/**
 * @brief Change the current subtile in the UI.
 * @param subtitle
 */
void MediaPlayerView::onSubtitleChanged(const QString& subtitle)
{
    m_ui->statusLabel->setText(subtitle);
}

/**
 * @brief Show the QMediaPlayer error
 * @param error
 */
void MediaPlayerView::onErrorOccured(const QString& error)
{
    m_ui->playButton->setEnabled(false);
    m_ui->statusLabel->setText(error);
}

/**
 * @brief Called when the user wants to load a media file in the player.
 */
void MediaPlayerView::openFile()
{
    QFileDialog fileDialog(this);
    fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
    fileDialog.setWindowTitle(tr("Open Media File"));

    fileDialog.setDirectory(
        QStandardPaths::standardLocations(QStandardPaths::MoviesLocation).value(0, QDir::homePath()));
    if (fileDialog.exec() == QDialog::Accepted)
    {
        m_mediaPlayer->setMedia(fileDialog.selectedUrls().constFirst());
        m_ui->statusLabel->setText(QString());
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

}    // namespace View

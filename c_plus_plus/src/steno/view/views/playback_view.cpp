#include "playback_view.h"
#include "ui_playback_view.h"

#include "model/i_video_player.h"

#include <QtWidgets>

namespace View
{

PlaybackView::PlaybackView(/*Model::IVideoPlayer& videoPlayer,*/ QWidget *parent)
    : AbstractView("Playback", parent)
    , m_ui(new Ui::PlaybackView)
    , m_mediaPlayer(new QMediaPlayer(this, QMediaPlayer::VideoSurface))
{
    m_ui->setupUi(this);
    m_ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));

    connect(m_ui->openButton, &QAbstractButton::clicked, [=]{ openFile(); });
    connect(m_ui->playButton, &QAbstractButton::clicked, [=]{ play(); });
    connect(m_ui->positionSlider, &QAbstractSlider::sliderMoved, [=](int position){ setPosition(position); });

    m_mediaPlayer->setVideoOutput(m_ui->videoWidget);

    connect(m_mediaPlayer, &QMediaPlayer::stateChanged, [=](QMediaPlayer::State state){ mediaStateChanged(state); });
    connect(m_mediaPlayer, &QMediaPlayer::positionChanged, [=](qint64 position){ positionChanged(position); });
    connect(m_mediaPlayer, &QMediaPlayer::durationChanged, [=](qint64 duration){ durationChanged(duration); });
    connect(m_mediaPlayer, QOverload<QMediaPlayer::Error>::of(&QMediaPlayer::error), [=]{ handleError(); });
}

PlaybackView::~PlaybackView()
{
    delete m_ui;
}

void PlaybackView::openFile()
{
    QFileDialog fileDialog(this);
    fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
    fileDialog.setWindowTitle(tr("Open Media File"));

    QStringList supportedMimeTypes = m_mediaPlayer->supportedMimeTypes();
    if (!supportedMimeTypes.isEmpty())
    {
        fileDialog.setMimeTypeFilters(supportedMimeTypes);
    }

    fileDialog.setDirectory(QStandardPaths::standardLocations(QStandardPaths::MoviesLocation).value(0, QDir::homePath()));
    if (fileDialog.exec() == QDialog::Accepted)
    {
        setUrl(fileDialog.selectedUrls().constFirst());
    }
}

void PlaybackView::setUrl(const QUrl &url)
{
    m_ui->errorLabel->setText(QString());
    setWindowFilePath(url.isLocalFile() ? url.toLocalFile() : QString());
    m_mediaPlayer->setMedia(url);
    m_ui->playButton->setEnabled(true);
}

void PlaybackView::play()
{
    switch (m_mediaPlayer->state())
    {
        case QMediaPlayer::PlayingState:
            m_mediaPlayer->pause();
            break;
        default:
            m_mediaPlayer->play();
            break;
    }
}

void PlaybackView::mediaStateChanged(QMediaPlayer::State state)
{
    switch(state) {
    case QMediaPlayer::PlayingState:
        m_ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
        break;
    default:
        m_ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
        break;
    }
}

void PlaybackView::positionChanged(qint64 position)
{
    m_ui->positionSlider->setValue(static_cast<int>(position));
}

void PlaybackView::durationChanged(qint64 duration)
{
    m_ui->positionSlider->setRange(0, static_cast<int>(duration));
}

void PlaybackView::setPosition(int position)
{
    m_mediaPlayer->setPosition(position);
}

void PlaybackView::handleError()
{
    m_ui->playButton->setEnabled(false);
    const QString errorString = m_mediaPlayer->errorString();
    QString message = "Error: ";

    if (errorString.isEmpty())
    {
        message += " #" + QString::number(int(m_mediaPlayer->error()));
    }
    else
    {
        message += errorString;
    }

    m_ui->errorLabel->setText(message);
}

} // View

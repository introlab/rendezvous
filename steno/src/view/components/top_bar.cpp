#include "top_bar.h"
#include "colors.h"
#include "model/app_config.h"
#include "model/config/config.h"
#include "model/transcription/transcription_config.h"
#include "model/utils/filesutil.h"
#include "ui_top_bar.h"

#include <QDesktopServices>
#include <QNetworkReply>
#include <QStyle>
#include <QUrl>

namespace View
{
TopBar::TopBar(std::shared_ptr<Model::IStream> stream, std::shared_ptr<Model::Media> media,
               std::shared_ptr<Model::Transcription> transcription, std::shared_ptr<Model::Config> config,
               QWidget* parent)
    : QWidget(parent)
    , m_ui(new Ui::TopBar)
    , m_stream(stream)
    , m_media(media)
    , m_transcription(transcription)
    , m_config(config)
{
    m_ui->setupUi(this);

    m_ui->recordButton->setDisabled(true);

    QPalette pal = palette();
    pal.setColor(QPalette::Window, LIGHT_GREEN);
    pal.setColor(QPalette::WindowText, WHITE);
    pal.setColor(QPalette::Button, DARK_GREEN);
    pal.setColor(QPalette::ButtonText, WHITE);
    setAutoFillBackground(true);
    setPalette(pal);

    connect(m_stream.get(), &Model::IStream::stateChanged,
            [=](const Model::IStream::State& state) { onStreamStateChanged(state); });
    connect(m_ui->startButton, &QAbstractButton::clicked, [=] { onStartButtonClicked(); });

    connect(m_media.get(), &Model::Media::recorderStateChanged,
            [=](const QMediaRecorder::State& state) { onRecorderStateChanged(state); });
    connect(m_ui->recordButton, &QAbstractButton::clicked, [=] { onRecordButtonClicked(); });

    connect(m_ui->meetButton, &QAbstractButton::clicked, [=] { QDesktopServices::openUrl(m_rendezvousMeetUrl); });

    connect(m_transcription.get(), &Model::Transcription::finished, [=](bool isOK, QString reply) {
        if (!isOK)
        {
            qDebug() << reply;
        }
    });
}

void TopBar::onStreamStateChanged(const Model::IStream::State& state)
{
    switch (state)
    {
        case Model::IStream::Started:
            m_ui->startButton->setText("Stop");
            m_ui->startButton->setDisabled(false);
            m_ui->recordButton->setDisabled(false);
            break;
        case Model::IStream::Stopping:
            m_ui->startButton->setText("Stopping");
            m_ui->startButton->setDisabled(true);
            m_ui->recordButton->setDisabled(true);
            break;
        case Model::IStream::Stopped:
            m_ui->startButton->setText("Start");
            m_ui->startButton->setDisabled(false);
            m_ui->recordButton->setDisabled(true);
            break;
    }
}

void TopBar::onStartButtonClicked()
{
    m_ui->startButton->setDisabled(true);
    switch (m_stream->state())
    {
        case Model::IStream::Started:
        {
            QApplication::processEvents();
            // We use a signal blocker to avoid queued signals from clicks on the startButton when the UI is disabled
            // The signals are reenable when the blocker is out of scope.
            QSignalBlocker blocker(m_ui->startButton);
            m_stream->stop();
            break;
        }
        case Model::IStream::Stopping:
            break;
        case Model::IStream::Stopped:
            m_stream->start();
            break;
    }
}

void TopBar::onRecorderStateChanged(const QMediaRecorder::State& state)
{
    switch (state)
    {
        case QMediaRecorder::State::RecordingState:
            m_ui->recordButton->setText("Stop recording");
            break;
        case QMediaRecorder::State::StoppedState:
        {
            m_ui->recordButton->setText("Start recording");
            const auto transcriptionConfig = m_config->transcriptionConfig();
            const bool isTranscriptionEnabled =
                transcriptionConfig->value(Model::TranscriptionConfig::AUTOMATIC_TRANSCRIPTION).toBool();
            if (isTranscriptionEnabled)
            {
                QString folder = m_config->appConfig()->value(Model::AppConfig::OUTPUT_FOLDER).toString();
                QString lastRecordingPath;
                bool isOK = Model::Util::mostRecentModified(folder, "webm", lastRecordingPath);
                if (!isOK) break;

                m_transcription->transcribe(lastRecordingPath);
            }
            break;
        }
        case QMediaRecorder::State::PausedState:
            break;
    }
    m_ui->recordButton->setDisabled(false);
}

void TopBar::onRecordButtonClicked()
{
    m_ui->recordButton->setDisabled(true);
    switch (m_media->recorderState())
    {
        case QMediaRecorder::State::RecordingState:
            m_media->stopRecorder();
            break;
        case QMediaRecorder::State::StoppedState:
            m_media->startRecorder();
            break;
        case QMediaRecorder::State::PausedState:
            break;
    }
}

}    // namespace View

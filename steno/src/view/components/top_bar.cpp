#include "top_bar.h"
#include "ui_top_bar.h"

#include "colors.h"

#include <QDesktopServices>
#include <QSignalBlocker>
#include <QStyle>
#include <QUrl>

namespace View
{
TopBar::TopBar(std::shared_ptr<Model::IStream> stream, std::shared_ptr<Model::IRecorder> recorder, QWidget* parent)
    : QWidget(parent)
    , m_ui(new Ui::TopBar)
    , m_stream(stream)
    , m_recorder(recorder)
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

    connect(m_recorder.get(), &Model::IRecorder::stateChanged,
            [=](const Model::IRecorder::State& state) { onRecorderStateChanged(state); });
    connect(m_ui->recordButton, &QAbstractButton::clicked, [=] { onRecordButtonClicked(); });

    connect(m_ui->meetButton, &QAbstractButton::clicked,
            [] { QDesktopServices::openUrl(QUrl("https://rendezvous-meet.com/")); });
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

void TopBar::onRecorderStateChanged(const Model::IRecorder::State& state)
{
    switch (state)
    {
        case Model::IRecorder::Started:
            m_ui->recordButton->setText("Stop recording");
            break;
        case Model::IRecorder::Stopped:
            m_ui->recordButton->setText("Start recording");
            break;
    }
    m_ui->recordButton->setDisabled(false);
}

void TopBar::onRecordButtonClicked()
{
    m_ui->recordButton->setDisabled(true);
    switch (m_recorder->state())
    {
        case Model::IRecorder::Started:
            m_recorder->stop();
            break;
        case Model::IRecorder::Stopped:
            m_recorder->start();
            break;
    }
}

}    // namespace View

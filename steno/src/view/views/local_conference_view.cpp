#include "local_conference_view.h"
#include "ui_local_conference_view.h"

#include <QCameraViewfinder>
#include <QSignalBlocker>

namespace View
{
LocalConferenceView::LocalConferenceView(std::shared_ptr<Model::IStream> stream,
                                         std::shared_ptr<Model::IRecorder> recorder, QWidget* parent)
    : AbstractView("Local Conference", parent)
    , m_ui(new Ui::LocalConferenceView)
    , m_cameraViewfinder(new QCameraViewfinder(this))
    , m_stream(stream)
    , m_recorder(recorder)
{
    m_ui->setupUi(this);

    m_ui->virtualCameraLayout->addWidget(m_cameraViewfinder.get());
    m_cameraViewfinder->show();
    m_recorder->setCameraViewFinder(m_cameraViewfinder);

    connect(m_stream.get(), &Model::IStream::stateChanged,
            [=](const Model::IStream::State& state) { onStreamStateChanged(state); });
    connect(m_ui->startVirtualDevicesButton, &QAbstractButton::clicked, [=] { onStartVirtualDevicesButtonClicked(); });

    connect(m_recorder.get(), &Model::IRecorder::stateChanged,
            [=](const Model::IRecorder::State& state) { onRecorderStateChanged(state); });
    connect(m_ui->startRecorderButton, &QAbstractButton::clicked, [=] { onStartRecorderButtonClicked(); });

    m_ui->startRecorderButton->setDisabled(true);
}

void LocalConferenceView::onStartVirtualDevicesButtonClicked()
{
    m_ui->startVirtualDevicesButton->setDisabled(true);
    switch (m_stream->state())
    {
        case Model::IStream::Started:
        {
            QApplication::processEvents();
            // We use a signal blocker to avoid queued signals from clicks on the startButton when the UI is disabled
            // The signals are reenable when the blocker is out of scope.
            QSignalBlocker blocker(m_ui->startVirtualDevicesButton);
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

void LocalConferenceView::onStartRecorderButtonClicked()
{
    m_ui->startRecorderButton->setDisabled(true);
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

void LocalConferenceView::onStreamStateChanged(const Model::IStream::State& state)
{
    switch (state)
    {
        case Model::IStream::Started:
            m_ui->startVirtualDevicesButton->setText("Stop virtual devices");
            m_ui->startVirtualDevicesButton->setDisabled(false);
            m_ui->startRecorderButton->setDisabled(false);
            break;
        case Model::IStream::Stopping:
            m_ui->startVirtualDevicesButton->setText("Stopping virtual devices");
            m_ui->startVirtualDevicesButton->setDisabled(true);
            m_ui->startRecorderButton->setDisabled(true);
            break;
        case Model::IStream::Stopped:
            m_ui->startVirtualDevicesButton->setText("Start virtual devices");
            m_ui->startVirtualDevicesButton->setDisabled(false);
            m_ui->startRecorderButton->setDisabled(true);
            break;
    }
}

void LocalConferenceView::onRecorderStateChanged(const Model::IRecorder::State& state)
{
    switch (state)
    {
        case Model::IRecorder::Started:
            m_ui->startRecorderButton->setText("Stop recording");
            break;
        case Model::IRecorder::Stopped:
            m_ui->startRecorderButton->setText("Start recording");
            break;
    }
    m_ui->startRecorderButton->setDisabled(false);
}
}    // namespace View

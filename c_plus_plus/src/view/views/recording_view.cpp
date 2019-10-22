#include "recording_view.h"
#include "ui_recording_view.h"

#include <QCameraViewfinder>
#include <QUrl>

namespace View
{

RecordingView::RecordingView(QWidget *parent)
    : AbstractView("Recording", parent)
    , m_ui(new Ui::RecordingView)
    , m_recorder(new Model::Recorder(getCameraDevice(), getAudioDevice(), this))
    , m_cameraViewfinder(new QCameraViewfinder(this))
{
    m_ui->setupUi(this);
    m_ui->virtualCameraLayout->addWidget(m_cameraViewfinder);

    m_recorder->setCameraViewfinder(m_cameraViewfinder);
    m_cameraViewfinder->show();

    connect(m_ui->btnStartStopRecord, &QAbstractButton::clicked, [=]{ changeRecordButtonState(); });
}

void RecordingView::changeRecordButtonState()
{
    m_recordButtonState = !m_recordButtonState;

    if(m_recordButtonState)
    {
        m_ui->btnStartStopRecord->setText("Stop recording");
        m_recorder->start(getOutputPath());
    }
    else
    {
        m_ui->btnStartStopRecord->setText("Start recording");
        m_recorder->stop();
    }
}

void RecordingView::showEvent(QShowEvent */*event*/)
{
    m_recorder->startCamera();
}

void RecordingView::hideEvent(QHideEvent */*event*/)
{
    m_recorder->stopCamera();
}

QString RecordingView::getCameraDevice()
{
    return "/dev/video0";   //TODO: get device name from config file
}


QString RecordingView::getAudioDevice()
{
    return "default:";      //TODO: get device name from config file
}

QString RecordingView::getOutputPath()
{
    return "/home/walid/dev/workspace/";  //TODO: get path from settings
}

} // View

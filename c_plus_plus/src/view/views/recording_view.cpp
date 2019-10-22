#include "recording_view.h"
#include "ui_recording_view.h"

#include <QCamera>
#include <QCameraInfo>
#include <QCameraViewfinder>
#include <QListWidgetItem>

namespace View
{

RecordingView::RecordingView(QWidget *parent)
    : AbstractView("Recording", parent)
    , m_ui(new Ui::RecordingView)
    , m_camera(new QCamera(getCameraInfo()))
    , m_cameraViewfinder(new QCameraViewfinder(this))
{
    m_ui->setupUi(this);
    m_ui->virtualCameraLayout->addWidget(m_cameraViewfinder);

    m_camera->setViewfinder(m_cameraViewfinder);

    m_cameraViewfinder->show();

    connect(m_ui->btnStartStopRecord, &QAbstractButton::clicked, [=]{ changeRecordButtonState(); });
}

QCameraInfo RecordingView::getCameraInfo()
{
    QCameraInfo cameraInfo;
    QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
    if(cameras.length() == 1)
    {
        cameraInfo = QCameraInfo::defaultCamera();
    }
    else
    {
        foreach (const QCameraInfo &camInfo, cameras)
        {
            if (camInfo.deviceName() == "/dev/video1")   //TODO: get device from config file
                cameraInfo = camInfo;
        }
    }

    return cameraInfo;
}

void RecordingView::changeRecordButtonState()
{
    m_recordButtonState = !m_recordButtonState;

    if(m_recordButtonState)
        m_ui->btnStartStopRecord->setText("Stop recording");    // TODO: Call start on IRecorder
    else
        m_ui->btnStartStopRecord->setText("Start recording");   // TODO: Call stop on IRecorder
}

void RecordingView::showEvent(QShowEvent */*event*/)
{
    if(m_camera->state() != QCamera::State::ActiveState)
    {
        m_camera->start();
    }
}

void RecordingView::hideEvent(QHideEvent */*event*/)
{
    if(m_camera->state() == QCamera::State::ActiveState)
    {
        m_camera->stop();
    }
}

} // View

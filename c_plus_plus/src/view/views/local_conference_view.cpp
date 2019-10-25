#include "local_conference_view.h"
#include "ui_local_conference_view.h"

#include <QCamera>
#include <QCameraInfo>
#include <QCameraViewfinder>
#include <QListWidgetItem>

namespace View
{

LocalConferenceView::LocalConferenceView(QWidget *parent)
    : AbstractView("Local Conference", parent)
    , m_ui(new Ui::LocalConferenceView)
    , m_camera(new QCamera(getCameraInfo()))
    , m_cameraViewfinder(new QCameraViewfinder(this))
{
    m_ui->setupUi(this);
    m_ui->virtualCameraLayout->addWidget(m_cameraViewfinder);

    m_camera->setViewfinder(m_cameraViewfinder);

    m_cameraViewfinder->show();

    connect(m_ui->btnStartStopRecord, &QAbstractButton::clicked, [=]{ changeRecordButtonState(); });
}

QCameraInfo LocalConferenceView::getCameraInfo()
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

void LocalConferenceView::changeRecordButtonState()
{
    m_recordButtonState = !m_recordButtonState;

    if(m_recordButtonState)
        m_ui->btnStartStopRecord->setText("Stop recording");    // TODO: Call start on IRecorder
    else
        m_ui->btnStartStopRecord->setText("Start recording");   // TODO: Call stop on IRecorder
}

void LocalConferenceView::showEvent(QShowEvent */*event*/)
{
    if(m_camera->state() != QCamera::State::ActiveState)
    {
        m_camera->start();
    }
}

void LocalConferenceView::hideEvent(QHideEvent */*event*/)
{
    if(m_camera->state() == QCamera::State::ActiveState)
    {
        m_camera->stop();
    }
}

} // View

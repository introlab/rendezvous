#include "recording_view.h"
#include "ui_recording_view.h"

namespace View
{

RecordingView::RecordingView(QWidget *parent)
    : AbstractView("Recording", parent)
    , mUi(new Ui::RecordingView)
{
    mUi->setupUi(this);

    mCameraViewfinder = new QCameraViewfinder(this);
    mCameraViewfinder->show();

    mCamera = new QCamera(getCameraInfo());
    mCamera->setViewfinder(mCameraViewfinder);

    mUi->virtualCameraLayout->addWidget(mCameraViewfinder);

    connect(mUi->btnStartStopRecord, &QAbstractButton::clicked, [=]{ changeRecordButtonState(); });
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
    mRecordButtonState = !mRecordButtonState;

    if(mRecordButtonState)
        mUi->btnStartStopRecord->setText("Stop recording");    // TODO: Call start on IRecorder
    else
        mUi->btnStartStopRecord->setText("Start recording");   // TODO: Call stop on IRecorder
}

void RecordingView::showEvent(QShowEvent *event)
{
    if(mCamera->state() != QCamera::State::ActiveState)
    {
        mCamera->start();
    }
}

void RecordingView::hideEvent(QHideEvent *event)
{
    if(mCamera->state() == QCamera::State::ActiveState)
    {
        mCamera->stop();
    }
}

} // View

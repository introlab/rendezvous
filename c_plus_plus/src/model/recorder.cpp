#include "recorder.h"

#include <QCamera>
#include <QCameraInfo>
#include <QCameraViewfinder>
#include <QMediaRecorder>
#include <QUrl>

namespace Model
{

Recorder::Recorder(const QString cameraDevice, QWidget *parent)
    : IRecorder(parent)
    , m_camera(new QCamera(getCameraInfo(cameraDevice), this))
    , m_mediaRecorder(new QMediaRecorder(m_camera, this))
{
    m_camera->setCaptureMode(QCamera::CaptureMode::CaptureVideo);

    QVideoEncoderSettings videoSettings;
    videoSettings.setQuality(QMultimedia::VeryHighQuality);
    videoSettings.setCodec("video/x-vp8");
    m_mediaRecorder->setVideoSettings(videoSettings);

    QAudioEncoderSettings audioSettings;
    audioSettings.setQuality(QMultimedia::VeryHighQuality);
    audioSettings.setCodec("audio/x-vorbis");
    m_mediaRecorder->setAudioSettings(audioSettings);

    m_mediaRecorder->setContainerFormat("video/webm");
}

void Recorder::start(const QString outputPath)
{
    m_mediaRecorder->setOutputLocation(QUrl::fromLocalFile(outputPath + "/media"));

    if(m_mediaRecorder->isAvailable() && m_camera->status() == QCamera::Status::ActiveStatus)
    {
        m_mediaRecorder->record();
    }
}

void Recorder::stop()
{
    m_mediaRecorder->stop();
}

QCameraInfo Recorder::getCameraInfo(QString cameraDevice)
{
    QCameraInfo defaultCameraInfo = QCameraInfo::defaultCamera();

    if(cameraDevice != "")
    {
        QList<QCameraInfo> cameras = QCameraInfo::availableCameras();

        foreach (const QCameraInfo &cameraInfo, cameras)
        {
            if (cameraInfo.deviceName() == cameraDevice)
                return cameraInfo;
        }
    }

    return defaultCameraInfo;
}

void Recorder::startCamera()
{
    if(m_camera->state() != QCamera::State::ActiveState)
    {
        m_camera->start();
    }
}

void Recorder::stopCamera()
{
    if(m_camera->state() == QCamera::State::ActiveState)
    {
        m_camera->stop();
    }
}

void Recorder::setCameraViewfinder(QCameraViewfinder *viewfinder)
{
    m_camera->setViewfinder(viewfinder);
}

} // Model

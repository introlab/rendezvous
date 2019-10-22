#include "recorder.h"

#include <QMediaRecorder>
#include <QAudioRecorder>
#include <QCamera>
#include <QCameraInfo>
#include <QCameraViewfinder>
#include <QUrl>
#include <QVideoEncoderSettings>

namespace Model
{

Recorder::Recorder(const QString cameraDevice, const QString audioDevice, QWidget *parent)
    : IRecorder(parent)
    , m_camera(new QCamera(getCameraInfo(cameraDevice)))
    , m_audioRecorder(new QAudioRecorder(this))
{
    m_camera->setCaptureMode(QCamera::CaptureMode::CaptureVideo);

    QVideoEncoderSettings videoSettings;
    videoSettings.setCodec("video/x-h264");
    videoSettings.setQuality(QMultimedia::VeryHighQuality);

    m_mediaRecorder = new QMediaRecorder(m_camera);
    m_mediaRecorder->setVideoSettings(videoSettings);
    m_mediaRecorder->setContainerFormat("avi");

    m_audioRecorder->setContainerFormat("audio/x-wav");
    setAudioInput(audioDevice);
}

QCameraInfo Recorder::getCameraInfo(QString cameraDevice)
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
            if (camInfo.deviceName() == cameraDevice)
                cameraInfo = camInfo;
        }
    }

    return cameraInfo;
}

void Recorder::start(const QString outputPath)
{
    QUrl videoPath = QUrl::fromLocalFile(outputPath + "video");
    QUrl audioPath = QUrl::fromLocalFile(outputPath + "audio");

    m_mediaRecorder->setOutputLocation(videoPath);
    m_audioRecorder->setOutputLocation(audioPath);

    if(m_mediaRecorder->isAvailable() && m_camera->status() == QCamera::Status::ActiveStatus)
    {
        m_mediaRecorder->record();
        m_audioRecorder->record();
    }
}

void Recorder::stop()
{
    m_mediaRecorder->stop();
    m_audioRecorder->stop();
}

void Recorder::setAudioInput(const QString audioDevice)
{
    QStringList inputs = m_audioRecorder->audioInputs();
    QString selectedInput = m_audioRecorder->defaultAudioInput();

    foreach (QString input, inputs)
    {
        if(input == audioDevice)
            selectedInput = input;
    }

    m_audioRecorder->setAudioInput(selectedInput);
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

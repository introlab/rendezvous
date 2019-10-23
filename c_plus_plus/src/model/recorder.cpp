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
    , m_audioRecorder(new QAudioRecorder())
{
    m_camera->setCaptureMode(QCamera::CaptureMode::CaptureVideo);

    // Video
    QVideoEncoderSettings videoSettings;
    videoSettings.setCodec("video/x-h264");
    videoSettings.setQuality(QMultimedia::VeryHighQuality);

    m_mediaRecorder = new QMediaRecorder(m_camera);
    m_mediaRecorder->setVideoSettings(videoSettings);
    m_mediaRecorder->setContainerFormat("avi");

    // Audio
    QAudioEncoderSettings audioSettings;
    audioSettings.setSampleRate(16000);
    audioSettings.setQuality(QMultimedia::VeryHighQuality);
    m_audioRecorder->setAudioSettings(audioSettings);

    m_audioRecorder->setContainerFormat("audio/x-wav");
    m_audioRecorder->setAudioInput(getAudioInput(audioDevice));
}

void Recorder::start(const QString outputPath)
{
    QUrl videoPath = QUrl::fromLocalFile(outputPath + "/video");
    QUrl audioPath = QUrl::fromLocalFile(outputPath + "/audio");

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

QString Recorder::getAudioInput(const QString audioDevice)
{
    QString defaultInput = m_audioRecorder->defaultAudioInput();

    if(audioDevice != "")
    {
        QStringList inputs = m_audioRecorder->audioInputs();

        foreach (QString input, inputs)
        {
            if(input == audioDevice)
                return input;
        }
    }

    return defaultInput;
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

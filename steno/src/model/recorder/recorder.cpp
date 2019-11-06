#include "recorder.h"

#include "model/app_config.h"
#include "model/app_constants.h"

#include <QUrl>

namespace Model
{
Recorder::Recorder(std::shared_ptr<Config> settings, QWidget *parent)
    : IRecorder(parent)
    , m_state(IRecorder::State::Stopped)
    , m_camera(cameraInfo(), this)
    , m_mediaRecorder(&m_camera)
    , m_settings(settings)
{
    m_camera.setCaptureMode(QCamera::CaptureVideo);

    QVideoEncoderSettings videoSettings;
    videoSettings.setQuality(QMultimedia::VeryHighQuality);
    videoSettings.setCodec("video/x-vp8");
    m_mediaRecorder.setVideoSettings(videoSettings);

    QAudioEncoderSettings audioSettings;
    audioSettings.setQuality(QMultimedia::VeryHighQuality);
    audioSettings.setCodec("audio/x-vorbis");
    m_mediaRecorder.setAudioSettings(audioSettings);

    m_mediaRecorder.setContainerFormat("video/webm");

    connect(&m_camera, &QCamera::statusChanged, [=](QCamera::Status status) { onCameraStatusChanged(status); });
}

Recorder::~Recorder()
{
    stopCamera();
}

void Recorder::start()
{
    if (m_state != IRecorder::State::Started)
    {
        startCamera();
    }
}

void Recorder::stop()
{
    if (m_state != IRecorder::State::Stopped)
    {
        m_mediaRecorder.stop();
        stopCamera();
        updateState(IRecorder::State::Stopped);
    }
}

void Recorder::setCameraViewFinder(std::shared_ptr<QCameraViewfinder> cameraViewFinder)
{
    m_camera.setViewfinder(cameraViewFinder.get());
}

void Recorder::onCameraStatusChanged(QCamera::Status status)
{
    switch (status)
    {
        case QCamera::Status::ActiveStatus:
            m_mediaRecorder.setOutputLocation(
                m_settings->subConfig(Model::Config::APP)->value(Model::AppConfig::OUTPUT_FOLDER).toString());
            m_mediaRecorder.record();
            updateState(IRecorder::State::Started);
            break;
        default:
            break;
    }
}

void Recorder::updateState(const IRecorder::State &state)
{
    m_state = state;
    emit stateChanged(m_state);
}

QCameraInfo Recorder::cameraInfo()
{
    QCameraInfo defaultCameraInfo = QCameraInfo::defaultCamera();

    if (!Model::VIRTUAL_CAMERA_DEVICE.isEmpty())
    {
        QList<QCameraInfo> cameras = QCameraInfo::availableCameras();

        foreach (const QCameraInfo &cameraInfo, cameras)
        {
            if (cameraInfo.deviceName() == Model::VIRTUAL_CAMERA_DEVICE) return cameraInfo;
        }
    }

    return defaultCameraInfo;
}

void Recorder::startCamera()
{
    if (m_camera.state() != QCamera::State::ActiveState)
    {
        m_camera.start();
    }
}

void Recorder::stopCamera()
{
    if (m_camera.state() == QCamera::State::ActiveState)
    {
        m_camera.stop();
    }
}

}    // namespace Model

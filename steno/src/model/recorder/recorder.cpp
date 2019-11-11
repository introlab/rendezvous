#include "recorder.h"

#include "model/app_config.h"
#include "model/stream/video/video_config.h"

#include <QUrl>

namespace Model
{
Recorder::Recorder(std::shared_ptr<Config> config, QWidget *parent)
    : IRecorder(parent)
    , m_config(config)
    , m_state(IRecorder::State::Stopped)
    , m_camera(getCameraDeviceName(), this)
    , m_mediaRecorder(&m_camera)
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
                m_config->subConfig(Config::APP)->value(AppConfig::OUTPUT_FOLDER).toString());
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

QByteArray Recorder::getCameraDeviceName()
{
    QString deviceName =
        m_config->subConfig(Config::Group::VIDEO_OUTPUT)->value(VideoConfig::Key::DEVICE_NAME).toString();
    return deviceName.toUtf8();
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

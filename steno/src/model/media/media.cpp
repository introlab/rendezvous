#include "media.h"

#include "model/app_config.h"
#include "model/stream/video/video_config.h"

#include <QCameraInfo>
#include <QUrl>

namespace Model
{
Media::Media(std::shared_ptr<Config> config, std::shared_ptr<IStream> stream)
    : m_appConfig(config->subConfig(Config::APP))
    , m_videoConfig(config->subConfig(Config::VIDEO_OUTPUT))
    , m_stream(stream)
{
    initCamera();
    initRecorder();

    connect(m_stream.get(), &IStream::stateChanged, [=](const IStream::State& state) { onStreamStateChanged(state); });
}

void Media::unLoadCamera()
{
    if (m_camera != nullptr)
    {
        m_camera->unload();
    }
}

/**
 * @brief Tell the camera where to render the camera images.
 * @param [IN] view
 */
void Media::setViewFinder(QCameraViewfinder* view)
{
    if (!m_camera.isNull() && view != nullptr)
    {
        m_camera->unload();
        m_camera->setViewfinder(view);
        view->show();
        m_camera->start();
    }
}

/**
 * @brief Start the recording of the audio and video stream.
 */
void Media::startRecorder()
{
    if (m_mediaRecorder)
    {
        m_mediaRecorder->record();
    }
}

/**
 * @brief Stop the recording of the audio and video stream.
 */
void Media::stopRecorder()
{
    if (m_mediaRecorder)
    {
        m_mediaRecorder->stop();
    }
}

/**
 * @brief Media::recorderState
 * @return the current state of the Qt recording object.
 */
QMediaRecorder::State Media::recorderState() const
{
    return m_mediaRecorder->state();
}

/**
 * @brief Callback when the stream change of state.
 * @param [IN] state
 */
void Media::onStreamStateChanged(const IStream::State& state)
{
    switch (state)
    {
        case IStream::Started:
            break;
        case IStream::Stopping:
            stopRecorder();
            break;
        case IStream::Stopped:
            break;
    }
}

/**
 * @brief Initialize the camera to record based on the desired camera in the config file.
 */
void Media::initCamera()
{
    auto deviceName = m_videoConfig->value(VideoConfig::Key::DEVICE_NAME).toString();
    for (auto cameraInfo : QCameraInfo::availableCameras())
    {
        if (cameraInfo.deviceName() == deviceName)
        {
            m_camera.reset(new QCamera(cameraInfo));
            m_camera->setCaptureMode(QCamera::CaptureMode::CaptureVideo);
            break;
        }
    }
}

/**
 * Initialization of the Qt recording object (Codecs, qualities, etc.)
 */
void Media::initRecorder()
{
    m_mediaRecorder.reset(new QMediaRecorder(m_camera.get()));

    QVideoEncoderSettings videoSettings;
    videoSettings.setQuality(QMultimedia::VeryHighQuality);
    videoSettings.setCodec("video/x-vp8");
    m_mediaRecorder->setVideoSettings(videoSettings);

    QAudioEncoderSettings audioSettings;
    audioSettings.setQuality(QMultimedia::VeryHighQuality);
    audioSettings.setCodec("audio/x-vorbis");
    m_mediaRecorder->setAudioSettings(audioSettings);

    m_mediaRecorder->setContainerFormat("video/webm");

    m_mediaRecorder->setOutputLocation(m_appConfig->value(AppConfig::OUTPUT_FOLDER).toString());
    connect(m_mediaRecorder.get(), &QMediaRecorder::stateChanged,
            [=](QMediaRecorder::State state) { emit recorderStateChanged(state); });

    connect(m_mediaRecorder.get(), QOverload<QMediaRecorder::Error>::of(&QMediaRecorder::error),
            [=](const QMediaRecorder::Error& error) { qCritical() << "Media Recorder Error: " << error; });
}

}    // namespace Model

#include "media.h"

#include "model/stream/video/video_config.h"
#include "model/app_config.h"

#include <QCameraInfo>
#include <QUrl>

namespace Model
{
Media::Media(std::shared_ptr<Config> config)
    : m_appConfig(config->subConfig(Config::APP))
    , m_videoConfig(config->subConfig(Config::VIDEO_OUTPUT))
{
    initCamera();
    initRecorder();
}

void Media::setViewFinder(QCameraViewfinder* view)
{
    m_camera->unload();
    m_camera->setViewfinder(view);
    m_camera->start();
}

void Media::startRecorder()
{
    m_mediaRecorder->record();
}

void Media::stopRecorder()
{
    m_mediaRecorder->stop();
}

QMediaRecorder::State Media::recorderState() const
{
    return m_mediaRecorder->state();
}

void Media::initCamera()
{
    auto deviceName = m_videoConfig->value(VideoConfig::Key::DEVICE_NAME).toString();
    for (auto cameraInfo : QCameraInfo::availableCameras())
    {
        if (cameraInfo.deviceName() == deviceName)
        {
            m_camera.reset(new QCamera(cameraInfo));

            auto imageProcessing = m_camera->imageProcessing();

            if (imageProcessing->isAvailable()) {
                imageProcessing->setContrast(1);
            }

            break;
        }
    }
}

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
   connect(m_mediaRecorder.get(), &QMediaRecorder::stateChanged, [=](QMediaRecorder::State state){ emit recorderStateChanged(state); });
}

}

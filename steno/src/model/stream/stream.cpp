#include "stream.h"

#include "model/stream/audio/file/raw_file_audio_sink.h"
#include "model/stream/audio/odas/odas_audio_source.h"
#include "model/stream/audio/odas/odas_client.h"
#include "model/stream/audio/odas/odas_position_source.h"
#include "model/stream/audio/pulseaudio/pulseaudio_sink.h"
#include "model/stream/utils/images/images.h"
#include "model/stream/utils/math/angle_calculations.h"
#include "model/stream/utils/models/spherical_angle_rect.h"
#include "model/stream/utils/threads/lock_triple_buffer.h"
#include "model/stream/utils/threads/readerwriterqueue.h"
#include "model/stream/video/impl/implementation_factory.h"
#include "model/stream/video/output/virtual_camera_output.h"

#include <string>
#include <vector>

#include <QCoreApplication>

namespace Model
{
Stream::Stream(const VideoConfig& videoInputConfig, const VideoConfig& videoOutputConfig,
               const AudioConfig& audioInputConfig, const AudioConfig& audioOutputConfig,
               const DewarpingConfig& dewarpingConfig)
    : m_state(IStream::State::Stopped)
    , m_videoInputConfig(videoInputConfig)
    , m_videoOutputConfig(videoOutputConfig)
    , m_audioInputConfig(audioInputConfig)
    , m_audioOutputConfig(audioOutputConfig)
    , m_dewarpingConfig(dewarpingConfig)
    , m_mediaThread(nullptr)
    , m_detectionThread(nullptr)
    , m_implementationFactory(false)
{
    int detectionDewarpingCount = 4;
    float aspectRatio = 3.f / 4.f;
    float minElevation = math::deg2rad(0.f);
    float maxElevation = math::deg2rad(90.f);

    m_imageBuffer = std::make_shared<LockTripleBuffer<Image>>(RGBImage(m_videoInputConfig.resolution));
    std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>> detectionQueue =
        std::make_shared<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>>(1);

    std::string configFile =
        (QCoreApplication::applicationDirPath() + "/../configs/yolo/cfg/yolov3-tiny.cfg").toStdString();
    std::string weightsFile =
        (QCoreApplication::applicationDirPath() + "/../configs/yolo/weights/yolov3-tiny.weights").toStdString();
    std::string metaFile = (QCoreApplication::applicationDirPath() + "/../configs/yolo/cfg/coco.data").toStdString();

    m_objectFactory = m_implementationFactory.getDetectionObjectFactory();
    m_objectFactory->allocateObjectLockTripleBuffer(*m_imageBuffer);

    m_detectionThread = std::make_unique<DetectionThread>(
        m_imageBuffer, m_implementationFactory.getDetector(configFile, weightsFile, metaFile), detectionQueue,
        m_implementationFactory.getDetectionFisheyeDewarper(aspectRatio),
        m_implementationFactory.getDetectionObjectFactory(), m_implementationFactory.getDetectionSynchronizer(),
        m_dewarpingConfig, detectionDewarpingCount);

    m_mediaThread = std::make_unique<MediaThread>(
        std::make_unique<OdasAudioSource>(10030, 1000 / m_videoOutputConfig.fpsTarget, 4, m_audioInputConfig),
        std::make_unique<RawFileAudioSink>("audio_output.raw"), std::make_unique<OdasPositionSource>(10020),
        m_implementationFactory.getCameraReader(m_videoInputConfig), m_implementationFactory.getFisheyeDewarper(),
        m_implementationFactory.getObjectFactory(), std::make_unique<VirtualCameraOutput>(m_videoOutputConfig),
        m_implementationFactory.getSynchronizer(),
        std::make_unique<VirtualCameraManager>(aspectRatio, minElevation, maxElevation), detectionQueue, m_imageBuffer,
        m_implementationFactory.getImageConverter(), m_dewarpingConfig, m_videoInputConfig, m_videoOutputConfig,
        m_audioInputConfig, m_audioOutputConfig);

    m_odasClient = std::make_unique<OdasClient>();
    m_odasClient->attach(this);
}

Stream::~Stream()
{
    m_objectFactory->deallocateObjectLockTripleBuffer(*m_imageBuffer);
}

void Stream::start()
{
    if (m_state == IStream::State::Stopped)
    {
        m_mediaThread->start();
        m_detectionThread->start();
        m_odasClient->start();
        updateState(IStream::State::Started);
    }
}

void Stream::stop()
{
    updateState(IStream::State::Stopping);

    if (m_odasClient->getState() != OdasClientState::CRASHED)
    {
        m_odasClient->stop();
        m_odasClient->join();
    }

    m_detectionThread->stop();
    m_detectionThread->join();
    m_mediaThread->stop();
    m_mediaThread->join();

    updateState(IStream::State::Stopped);
}

void Stream::updateState(const IStream::State& state)
{
    m_state = state;
    emit stateChanged(m_state);
}

void Stream::updateObserver()
{
    OdasClientState state = m_odasClient->getState();
    if (state == OdasClientState::CRASHED)
    {
        stop();
    }
}
}    // namespace Model

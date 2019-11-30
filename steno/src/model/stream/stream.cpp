#include "stream.h"

#include "model/app_config.h"
#include "model/stream/audio/audio_config.h"
#include "model/stream/audio/file/raw_file_audio_sink.h"
#include "model/stream/audio/odas/odas_audio_source.h"
#include "model/stream/audio/odas/odas_client.h"
#include "model/stream/audio/odas/odas_position_source.h"
#include "model/stream/audio/pulseaudio/pulseaudio_sink.h"
#include "model/stream/stream_config.h"
#include "model/stream/utils/images/images.h"
#include "model/stream/utils/models/dim2.h"
#include "model/stream/utils/models/spherical_angle_rect.h"
#include "model/stream/utils/threads/lock_triple_buffer.h"
#include "model/stream/utils/threads/readerwriterqueue.h"
#include "model/stream/video/detection/darknet_config.h"
#include "model/stream/video/dewarping/models/dewarping_config.h"
#include "model/stream/video/impl/implementation_factory.h"
#include "model/stream/video/output/default_virtual_camera_output.h"
#include "model/stream/video/output/virtual_camera_output.h"
#include "model/stream/video/video_config.h"

#include <string>
#include <vector>

#include <QCoreApplication>

namespace Model
{
Stream::Stream(std::shared_ptr<Config> config)
    : m_state(IStream::State::Stopped)
    , m_mediaThread(nullptr)
    , m_detectionThread(nullptr)
    , m_config(config)
    , m_implementationFactory(false)
{
    std::shared_ptr<AudioConfig> audioInputConfig = m_config->audioInputConfig();
    std::shared_ptr<AudioConfig> audioOutputConfig = m_config->audioOutputConfig();
    std::shared_ptr<VideoConfig> videoOutputConfig = m_config->videoOutputConfig();
    std::shared_ptr<VideoConfig> videoInputConfig = m_config->videoInputConfig();
    std::shared_ptr<StreamConfig> streamConfig = m_config->streamConfig();
    std::shared_ptr<DarknetConfig> darknetConfig = m_config->darknetConfig();

    float aspectRatio = streamConfig->value(StreamConfig::ASPECT_RATIO_WIDTH).toFloat() /
                        streamConfig->value(StreamConfig::ASPECT_RATIO_HEIGHT).toFloat();
    float minElevation = streamConfig->value(StreamConfig::MIN_ELEVATION).toFloat();
    float maxElevation = streamConfig->value(StreamConfig::MAX_ELEVATION).toFloat();
    Dim2<int> resolution(videoInputConfig->value(VideoConfig::WIDTH).toInt(),
                         videoInputConfig->value(VideoConfig::HEIGHT).toInt());

    int fps = videoOutputConfig->value(VideoConfig::FPS).toInt();
    if (fps == 0)
    {
        throw std::invalid_argument("Error in Stream - Fps cannot be 0");
    }
    int audioChunkDurationMs = 1000 / fps;

    // TODO: config
    int numberOfAudioBuffers = 5;
    int positionBufferSize = 1024;
    int odasAudioPort = 10030;
    int odasPositionPort = 10020;

    std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>> detectionQueue =
        std::make_shared<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>>(1);

    int sleepBetweenLayersForwardUs = darknetConfig->value(DarknetConfig::SLEEP_BETWEEN_LAYERS_FORWARD_US).toInt();
    std::string configFile =
        (QCoreApplication::applicationDirPath() + "/../configs/yolo/cfg/yolov3-tiny.cfg").toStdString();
    std::string weightsFile =
        (QCoreApplication::applicationDirPath() + "/../configs/yolo/weights/yolov3-tiny.weights").toStdString();
    std::string metaFile = (QCoreApplication::applicationDirPath() + "/../configs/yolo/cfg/coco.data").toStdString();

    m_imageBuffer = std::make_shared<LockTripleBuffer<Image>>(RGBImage(resolution));

    m_objectFactory = m_implementationFactory.getDetectionObjectFactory();
    m_objectFactory->allocateObjectLockTripleBuffer(*m_imageBuffer);

    m_detectionThread = std::make_unique<DetectionThread>(
        m_imageBuffer,
        m_implementationFactory.getDetector(configFile, weightsFile, metaFile, sleepBetweenLayersForwardUs),
        detectionQueue, m_implementationFactory.getDetectionFisheyeDewarper(aspectRatio),
        m_implementationFactory.getDetectionObjectFactory(), m_implementationFactory.getDetectionSynchronizer(),
        m_config->dewarpingConfig());

    m_mediaThread = std::make_unique<MediaThread>(
        std::make_unique<OdasAudioSource>(odasAudioPort, audioChunkDurationMs, numberOfAudioBuffers, audioInputConfig),
        std::make_unique<PulseAudioSink>(audioOutputConfig),
        std::make_unique<OdasPositionSource>(odasPositionPort, positionBufferSize),
        m_implementationFactory.getCameraReader(videoInputConfig), m_implementationFactory.getFisheyeDewarper(),
        m_implementationFactory.getObjectFactory(), std::make_unique<VirtualCameraOutput>(videoOutputConfig),
        m_implementationFactory.getSynchronizer(),
        std::make_unique<VirtualCameraManager>(aspectRatio, minElevation, maxElevation), detectionQueue, m_imageBuffer,
        m_implementationFactory.getImageConverter(), m_config);

    m_odasClient = std::make_unique<OdasClient>(m_config->appConfig());
    m_odasClient->attach(this);
}

Stream::~Stream()
{
    m_objectFactory->deallocateObjectLockTripleBuffer(*m_imageBuffer);
}

/**
 * @brief Start children threads.
 */
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

/**
 * @brief Stop children threads.
 */
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

/**
 * @brief Wait that all children threads terminate, do not call if you called Stream::stop
 */
void Stream::join()
{
    m_odasClient->join();
    m_detectionThread->join();
    m_mediaThread->join();
}

void Stream::updateState(const IStream::State& state)
{
    m_state = state;
    emit stateChanged(m_state);
}

/**
 * @brief What to do when a child thread crash.
 */
void Stream::updateObserver()
{
    OdasClientState state = m_odasClient->getState();
    if (state == OdasClientState::CRASHED)
    {
        stop();
    }
}
}    // namespace Model

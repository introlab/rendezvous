#include "stream.h"

#include "model/stream/audio/odas/odas_audio_source.h"
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
    , videoInputConfig_(videoInputConfig)
    , videoOutputConfig_(videoOutputConfig)
    , audioInputConfig_(audioInputConfig)
    , audioOutputConfig_(audioOutputConfig)
    , dewarpingConfig_(dewarpingConfig)
    , mediaThread_(nullptr)
    , detectionThread_(nullptr)
    , implementationFactory_(false)
{
    int detectionDewarpingCount = 4;
    float aspectRatio = 3.f / 4.f;
    float minElevation = math::deg2rad(0.f);
    float maxElevation = math::deg2rad(90.f);

    imageBuffer_ = std::make_shared<LockTripleBuffer<Image>>(RGBImage(videoInputConfig_.resolution));
    std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>> detectionQueue =
        std::make_shared<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>>(1);

    std::string configFile =
        (QCoreApplication::applicationDirPath() + "/../configs/yolo/cfg/yolov3-tiny.cfg").toStdString();
    std::string weightsFile =
        (QCoreApplication::applicationDirPath() + "/../configs/yolo/weights/yolov3-tiny.weights").toStdString();
    std::string metaFile = (QCoreApplication::applicationDirPath() + "/../configs/yolo/cfg/coco.data").toStdString();

    objectFactory_ = implementationFactory_.getDetectionObjectFactory();
    objectFactory_->allocateObjectLockTripleBuffer(*imageBuffer_);

    detectionThread_ = std::make_unique<DetectionThread>(
        imageBuffer_, implementationFactory_.getDetector(configFile, weightsFile, metaFile), detectionQueue,
        implementationFactory_.getDetectionFisheyeDewarper(aspectRatio),
        implementationFactory_.getDetectionObjectFactory(), implementationFactory_.getDetectionSynchronizer(),
        dewarpingConfig_, detectionDewarpingCount);

    mediaThread_ = std::make_unique<MediaThread>(
        std::make_unique<OdasAudioSource>(10030), std::make_unique<PulseAudioSink>(audioOutputConfig_),
        std::make_unique<OdasPositionSource>(10020), implementationFactory_.getCameraReader(videoInputConfig_),
        implementationFactory_.getFisheyeDewarper(), implementationFactory_.getObjectFactory(),
        std::make_unique<VirtualCameraOutput>(videoOutputConfig_), implementationFactory_.getSynchronizer(),
        std::make_unique<VirtualCameraManager>(aspectRatio, minElevation, maxElevation), detectionQueue, imageBuffer_,
        implementationFactory_.getImageConverter(), dewarpingConfig_, videoInputConfig_, videoOutputConfig_,
        audioInputConfig_, audioOutputConfig_);
}

Stream::~Stream()
{
    objectFactory_->deallocateObjectLockTripleBuffer(*imageBuffer_);
}

void Stream::start()
{
    if (m_state != IStream::State::Started)
    {
        mediaThread_->start();
        detectionThread_->start();
        updateState(IStream::State::Started);
    }
}

void Stream::stop()
{
    if (m_state != IStream::Stopped)
    {
        detectionThread_->stop();
        detectionThread_->join();
        mediaThread_->stop();
        mediaThread_->join();
        updateState(IStream::State::Stopped);
    }
}

void Stream::updateState(const IStream::State& state)
{
    m_state = state;
    emit stateChanged(m_state);
}

}    // namespace Model

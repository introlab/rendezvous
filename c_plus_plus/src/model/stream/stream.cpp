#include "stream.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

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

namespace Model
{
Stream::Stream(const VideoConfig& videoInputConfig, const VideoConfig& videoOutputConfig,
               const AudioConfig& audioInputConfig, const AudioConfig& audioOutputConfig,
               const DewarpingConfig& dewarpingConfig)
    : videoInputConfig_(videoInputConfig)
    , videoOutputConfig_(videoOutputConfig)
    , audioInputConfig_(audioInputConfig)
    , audioOutputConfig_(audioOutputConfig)
    , dewarpingConfig_(dewarpingConfig)
    , mediaThread_(nullptr)
    , detectionThread_(nullptr)
{
    bool useZeroCopyIfSupported = false;
    int detectionDewarpingCount = 4;
    float aspectRatio = 3.f / 4.f;
    float minElevation = math::deg2rad(0.f);
    float maxElevation = math::deg2rad(90.f);

    imageBuffer_ = std::make_shared<LockTripleBuffer<Image>>(RGBImage(videoInputConfig_.resolution));
    std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>> detectionQueue =
        std::make_shared<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>>(1);

    std::string root;
    std::string rendezvousStr = "rendezvous";
    std::string path = std::getenv("PWD");
    std::size_t index = path.find(rendezvousStr);

    if (index != std::string::npos)
    {
        root = path.substr(0, index + rendezvousStr.size());
    }
    else
    {
        throw std::runtime_error("You must run the application from rendezvous repo");
    }

    std::string configFile = root + "/config/yolo/cfg/yolov3-tiny.cfg";
    std::string weightsFile = root + "/config/yolo/weights/yolov3-tiny.weights";
    std::string metaFile = root + "/config/yolo/cfg/coco.data";

    ImplementationFactory implementationFactory(useZeroCopyIfSupported);

    objectFactory_ = implementationFactory.getDetectionObjectFactory();
    objectFactory_->allocateObjectLockTripleBuffer(*imageBuffer_);

    detectionThread_ = std::make_unique<DetectionThread>(
        imageBuffer_, implementationFactory.getDetector(configFile, weightsFile, metaFile), detectionQueue,
        implementationFactory.getDetectionFisheyeDewarper(aspectRatio),
        implementationFactory.getDetectionObjectFactory(), implementationFactory.getDetectionSynchronizer(),
        dewarpingConfig_, detectionDewarpingCount);

    mediaThread_ = std::make_unique<MediaThread>(
        std::make_unique<OdasAudioSource>(10030), std::make_unique<PulseAudioSink>(audioOutputConfig_),
        std::make_unique<OdasPositionSource>(10020), implementationFactory.getCameraReader(videoInputConfig_),
        implementationFactory.getFisheyeDewarper(), implementationFactory.getObjectFactory(),
        std::make_unique<VirtualCameraOutput>(videoOutputConfig_), implementationFactory.getSynchronizer(),
        std::make_unique<VirtualCameraManager>(aspectRatio, minElevation, maxElevation), detectionQueue, imageBuffer_,
        implementationFactory.getImageConverter(), dewarpingConfig_, videoInputConfig_, videoOutputConfig_,
        audioInputConfig_, audioOutputConfig_);
}

Stream::~Stream()
{
    objectFactory_->deallocateObjectLockTripleBuffer(*imageBuffer_);
}

void Stream::start()
{
    mediaThread_->start();
    detectionThread_->start();
}

void Stream::stop()
{
    detectionThread_->stop();
    detectionThread_->join();
    mediaThread_->stop();
    mediaThread_->join();
}

}    // namespace Model
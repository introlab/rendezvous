#include "stream.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "model/stream/utils/images/images.h"
#include "model/stream/utils/math/angle_calculations.h"
#include "model/stream/utils/models/spherical_angle_rect.h"
#include "model/stream/utils/threads/lock_triple_buffer.h"
#include "model/stream/utils/threads/readerwriterqueue.h"
#include "model/stream/video/impl/implementation_factory.h"
#include "model/stream/video/output/virtual_camera_output.h"

namespace Model
{
Stream::Stream(const VideoConfig& inputConfig, const VideoConfig& outputConfig, const DewarpingConfig& dewarpingConfig)
    : inputConfig_(inputConfig)
    , outputConfig_(outputConfig)
    , dewarpingConfig_(dewarpingConfig)
    , videoThread_(nullptr)
    , detectionThread_(nullptr)
{
    bool useZeroCopyIfSupported = false;
    int detectionDewarpingCount = 4;
    float aspectRatio = 3.f / 4.f;
    float minElevation = math::deg2rad(0.f);
    float maxElevation = math::deg2rad(90.f);

    imageBuffer_ = std::make_shared<LockTripleBuffer<Image>>(Image(inputConfig_.resolution, inputConfig.imageFormat));
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

    videoThread_ = std::make_unique<VideoThread>(
        implementationFactory.getCameraReader(inputConfig), implementationFactory.getFisheyeDewarper(),
        implementationFactory.getObjectFactory(), std::make_unique<VirtualCameraOutput>(outputConfig_),
        implementationFactory.getSynchronizer(),
        std::make_unique<VirtualCameraManager>(aspectRatio, minElevation, maxElevation), detectionQueue, imageBuffer_,
        implementationFactory.getImageConverter(), dewarpingConfig_, inputConfig_, outputConfig_);
}

Stream::~Stream() { objectFactory_->deallocateObjectLockTripleBuffer(*imageBuffer_); }

void Stream::start()
{
    videoThread_->start();
    detectionThread_->start();
}

void Stream::stop()
{
    detectionThread_->stop();
    detectionThread_->join();
    videoThread_->stop();
    videoThread_->join();
}

}    // namespace Model
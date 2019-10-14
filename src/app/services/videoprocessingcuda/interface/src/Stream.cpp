#include "Stream.h"

#include <vector>
#include <iostream>
#include <cstdlib>
#include <string>

#include "impl/ImplementationFactory.h"
#include "utils/images/FileImageWriter.h"
#include "utils/images/Image.h"
#include "utils/math/AngleCalculations.h"
#include "utils/models/AngleRect.h"
#include "utils/threads/LockTripleBuffer.h"
#include "utils/threads/readerwriterqueue.h"
#include "streaming/input/VideoStreamMock.h"


Stream::Stream(const CameraConfig& cameraConfig, const DewarpingConfig& dewarpingConfig)
    : cameraConfig_(cameraConfig)
    , dewarpingConfig_(dewarpingConfig)
    , videoThread_(nullptr)
    , detectionThread_(nullptr)
{
    bool useZeroCopyIfSupported = false;
    int detectionDewarpingCount = 4;
    float aspectRatio = 3.f / 4.f;
    float minElevation = math::deg2rad(0.f);
    float maxElevation = math::deg2rad(90.f);

    std::shared_ptr<LockTripleBuffer<Image>> imageBuffer = 
            std::make_shared<LockTripleBuffer<Image>>(cameraConfig_.resolution);
    std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<AngleRect>>> detectionQueue = 
        std::make_shared<moodycamel::ReaderWriterQueue<std::vector<AngleRect>>>(1);

    std::string root;
    std::string rendezvousStr = "rendezvous";
    std::string path = std::getenv("PWD");
    std::size_t index = path.find(rendezvousStr);

    if (index >= 0)
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

    std::unique_ptr<IObjectFactory> objectFactory = implementationFactory.getDetectionObjectFactory();
    objectFactory->allocateObjectLockTripleBuffer(*imageBuffer);

    detectionThread_ = std::make_unique<DetectionThread>(imageBuffer,
                                                         implementationFactory.getDetector(configFile, weightsFile, metaFile),
                                                         detectionQueue,
                                                         implementationFactory.getDetectionFisheyeDewarper(aspectRatio),
                                                         implementationFactory.getDetectionObjectFactory(),
                                                         implementationFactory.getDetectionSynchronizer(),
                                                         dewarpingConfig_,
                                                         detectionDewarpingCount);

    videoThread_ = std::make_unique<VideoThread>(std::make_unique<VideoStreamMock>(root + "/src/app/services/videoprocessingcuda/res/fisheye.jpg"),
                                                 implementationFactory.getFisheyeDewarper(),
                                                 implementationFactory.getObjectFactory(),
                                                 std::make_unique<FileImageWriter>("res", "test"),
                                                 implementationFactory.getSynchronizer(),
                                                 std::make_unique<VirtualCameraManager>(aspectRatio, minElevation, maxElevation),
                                                 detectionQueue, imageBuffer, dewarpingConfig_, cameraConfig_);
}

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
#include <iostream>
#include <memory>

#include "detection/DetectionThread.h"
#include "impl/ImplementationFactory.h"
#include "streaming/input/VideoStreamMock.h"
#include "utils/math/AngleCalculations.h"
#include "utils/images/FileImageWriter.h"
#include "VideoThread.h"
#include "virtualcamera/VirtualCameraManager.h"
#include "streaming/input/CameraConfig.h"

int main(int argc, char *argv[])
{
    try
    {
        bool useZeroCopyIfSupported = false;
        int detectionDewarpingCount = 4;

        float inRadius = 400.f;
        float outRadius = 1400.f;
        float angleSpan = math::deg2rad(90.f);
        float topDistorsionFactor = 0.08f;
        float bottomDistorsionFactor = 0.f;
        float fisheyeAngle = math::deg2rad(220.f);
        float minElevation = math::deg2rad(0.f);
        float maxElevation = math::deg2rad(90.f);
        float aspectRatio = 3.f / 4.f;
        DewarpingConfig dewarpingConfig(inRadius, outRadius, angleSpan,topDistorsionFactor, bottomDistorsionFactor, fisheyeAngle);

        int width = 2880;
        int height = 2160;
        int channels = 3;
        int fpsTarget = 20;
        CameraConfig cameraConfig(width, height, channels, fpsTarget);

        std::shared_ptr<LockTripleBuffer<Image>> imageBuffer = 
            std::make_shared<LockTripleBuffer<Image>>(cameraConfig.resolution);
        std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<AngleRect>>> detectionQueue = 
            std::make_shared<moodycamel::ReaderWriterQueue<std::vector<AngleRect>>>(1);

        std::string configFile = "../../../../config/yolo/cfg/yolov3-tiny.cfg";
        std::string weightsFile = "../../../../config/yolo/weights/yolov3-tiny.weights"; 
        std::string metaFile = "../../../../config/yolo/cfg/coco.data";

        ImplementationFactory implementationFactory(useZeroCopyIfSupported);

        std::unique_ptr<IObjectFactory> objectFactory = implementationFactory.getDetectionObjectFactory();
        objectFactory->allocateObjectLockTripleBuffer(*imageBuffer);

        DetectionThread detectionThread(imageBuffer,
                                        implementationFactory.getDetector(configFile, weightsFile, metaFile),
                                        detectionQueue,
                                        implementationFactory.getDetectionFisheyeDewarper(aspectRatio),
                                        implementationFactory.getDetectionObjectFactory(),
                                        implementationFactory.getDetectionSynchronizer(),
                                        dewarpingConfig,
                                        detectionDewarpingCount);

        VideoThread videoThread(std::make_unique<VideoStreamMock>("/home/mathieu/dev/workspace/rendezvous/src/app/services/videoprocessingcuda/res/fisheye.jpg"),
                                implementationFactory.getFisheyeDewarper(),
                                implementationFactory.getObjectFactory(),
                                std::make_unique<FileImageWriter>("res", "test"),
                                implementationFactory.getSynchronizer(),
                                std::make_unique<VirtualCameraManager>(aspectRatio, minElevation, maxElevation),
                                detectionQueue, imageBuffer, dewarpingConfig, cameraConfig);

        videoThread.start();
        detectionThread.start();

        videoThread.join();
        detectionThread.stop();
        detectionThread.join();

        objectFactory->deallocateObjectLockTripleBuffer(*imageBuffer);
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}

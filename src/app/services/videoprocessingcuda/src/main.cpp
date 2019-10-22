#include <iostream>
#include <memory>

#include "detection/DetectionThread.h"
#include "impl/ImplementationFactory.h"
#include "stream/input/CameraConfig.h"
#include "stream/input/CameraReader.h"
#include "utils/math/AngleCalculations.h"
#include "stream/output/VirtualCameraOutput.h"
#include "VideoThread.h"
#include "virtualcamera/VirtualCameraManager.h"

#include "stream/output/ImageFileWriter.h"
#include "stream/input/ImageFileReader.h"


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
        int fpsTarget = 10;
        CameraConfig cameraConfig(width, height, fpsTarget, "/dev/video2", ImageFormat::UYVY_FMT);

        std::shared_ptr<LockTripleBuffer<Image>> imageBuffer =
            std::make_shared<LockTripleBuffer<Image>>(RGBImage(cameraConfig.resolution));
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

        DetectionThread detectionThread(imageBuffer,
                                        implementationFactory.getDetector(configFile, weightsFile, metaFile),
                                        detectionQueue,
                                        implementationFactory.getDetectionFisheyeDewarper(aspectRatio),
                                        implementationFactory.getDetectionObjectFactory(),
                                        implementationFactory.getDetectionSynchronizer(),
                                        dewarpingConfig,
                                        detectionDewarpingCount);

        VideoThread videoThread(std::make_unique<CameraReader>(cameraConfig),
                                //std::make_unique<ImageFileReader>("res/fisheye.jpg", ImageFormat::UYVY_FMT),
                                implementationFactory.getFisheyeDewarper(),
                                implementationFactory.getObjectFactory(),
                                std::make_unique<VirtualCameraOutput>("/dev/video4", Dim2<int>(800, 600), ImageFormat::UYVY_FMT, fpsTarget),
                                //std::make_unique<ImageFileWriter>("res", "test"),
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

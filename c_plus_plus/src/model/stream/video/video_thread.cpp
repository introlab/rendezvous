#include "video_thread.h"

#include <iostream>

#include "model/stream/video/dewarping/dewarping_helper.h"
#include "model/stream/video/video_stabilizer.h"
#include "model/stream/utils/alloc/heap_object_factory.h"
#include "model/stream/utils/models/point.h"
#include "model/stream/video/virtualcamera/display_image_builder.h"

namespace Model
{

VideoThread::VideoThread(std::unique_ptr<IVideoInput> videoInput, std::unique_ptr<IFisheyeDewarper> dewarper,
                         std::unique_ptr<IObjectFactory> objectFactory, std::unique_ptr<IVideoOutput> videoOutput,
                         std::unique_ptr<ISynchronizer> synchronizer, std::unique_ptr<VirtualCameraManager> virtualCameraManager,
                         std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>> detectionQueue,
                         std::shared_ptr<LockTripleBuffer<Image>> imageBuffer, std::unique_ptr<IImageConverter> imageConverter,
                         const DewarpingConfig& dewarpingConfig, const VideoConfig& inputConfig, const VideoConfig& outputConfig)
    : videoInput_(std::move(videoInput))
    , dewarper_(std::move(dewarper))
    , objectFactory_(std::move(objectFactory))
    , videoOutput_(std::move(videoOutput))
    , synchronizer_(std::move(synchronizer))
    , virtualCameraManager_(std::move(virtualCameraManager))
    , detectionQueue_(detectionQueue)
    , imageBuffer_(imageBuffer)
    , imageConverter_(std::move(imageConverter))
    , dewarpingConfig_(dewarpingConfig)
    , inputConfig_(inputConfig)
    , outputConfig_(outputConfig)
{
    if (!videoInput_ || !dewarper_ || !objectFactory_ || !videoOutput_ || !synchronizer_ ||
        !virtualCameraManager_ || !detectionQueue_ || !imageBuffer_ || !imageConverter_)
    {
        throw std::invalid_argument("Error in VideoProcessorThread - Null is not a valid argument");
    } 
}

void VideoThread::run()
{
    try
    {
        HeapObjectFactory heapObjectFactory;
        DualBuffer<Image> displayBuffers(Image(outputConfig_.resolution, outputConfig_.imageFormat));
        DisplayImageBuilder displayImageBuilder(outputConfig_.resolution);

        heapObjectFactory.allocateObjectDualBuffer(displayBuffers);
        displayImageBuilder.setDisplayImageColor(displayBuffers.getCurrent());
        displayImageBuilder.setDisplayImageColor(displayBuffers.getInUse());

        Dim2<int> maxVcDim = displayImageBuilder.getMaxVirtualCameraDim();
        std::vector<Image> vcImages(5, RGBImage(maxVcDim.width, maxVcDim.height));
        objectFactory_->allocateObjectVector(vcImages);

        Point<float> fisheyeCenter(inputConfig_.resolution.width / 2.f, inputConfig_.resolution.height / 2.f);
        std::vector<SphericalAngleRect> detections;

        VideoStabilizer videoStabilizer(inputConfig_.fpsTarget);

        // Video loop start

        std::cout << "VideoThread loop started" << std::endl;

        while (!isAbortRequested())
        {
            videoStabilizer.startFrame();
            
            if (detectionQueue_->try_dequeue(detections))
            {
                virtualCameraManager_->updateVirtualCamerasGoal(detections);
            }

            virtualCameraManager_->updateVirtualCameras(videoStabilizer.getLastFrameTimeMs());
            
            const Image& rawImage = videoInput_->readImage();
            const Image& rgbImage = imageBuffer_->getCurrent();
            imageConverter_->convert(rawImage, rgbImage);
            imageBuffer_->swap();
            
            const std::vector<VirtualCamera> virtualCameras = virtualCameraManager_->getVirtualCameras();
            int vcCount = (int)virtualCameras.size();

            if (vcCount > 0)
            {
                // This should not happend often in theory, it's only if a large amount of virtual camera are required
                for (int i = vcImages.size(); i < vcCount; ++i)
                {
                    RGBImage vcImage(maxVcDim.width, maxVcDim.height);
                    objectFactory_->allocateObject(vcImage);
                    vcImages.push_back(vcImage);
                }

                Dim2<int> resizeDim(displayImageBuilder.getVirtualCameraDim(vcCount));
                std::vector<Image> vcResizeImages(vcCount, RGBImage(resizeDim));

                for (int i = 0; i < vcCount; ++i)
                {
                    const VirtualCamera& virtualCamera = virtualCameras[i];
                    Image& vcResizeImage = vcResizeImages[i];
                    vcResizeImage.hostData = vcImages[i].hostData;
                    vcResizeImage.deviceData = vcImages[i].deviceData;

                    DewarpingParameters vcParams = getDewarpingParametersFromAngleBoundingBox(virtualCamera, fisheyeCenter, dewarpingConfig_);
                    dewarper_->dewarpImageFiltered(rgbImage, vcResizeImage, vcParams);

                    // Ok, this is a hack until we work with raw data for dewarping of virtual cameras (convert on itself, only work from RGB)
                    Image inImage = vcResizeImage;
                    vcResizeImages[i] = Image(inImage.width, inImage.height, outputConfig_.imageFormat);
                    vcResizeImages[i].deviceData = inImage.deviceData;
                    vcResizeImages[i].hostData = inImage.hostData;
                    imageConverter_->convert(inImage, vcResizeImage);
                }

                const Image& displayImage = displayBuffers.getCurrent();
                displayImageBuilder.clearVirtualCamerasOnDisplayImage(displayImage);

                synchronizer_->sync();
                displayImageBuilder.createDisplayImage(vcResizeImages, displayImage);
                videoOutput_->writeImage(displayImage);
                displayBuffers.swap();
            }
            
            detections.clear();

            videoStabilizer.endFrame();
        }

        heapObjectFactory.deallocateObjectDualBuffer(displayBuffers);
        objectFactory_->deallocateObjectVector(vcImages);
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    std::cout << "VideoThread loop finished" << std::endl;
}
} // Model

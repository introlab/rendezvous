#include "VideoThread.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>

#ifndef NO_CUDA
    #include <cuda_runtime.h>
#endif

#include "utils/alloc/HeapObjectFactory.h"
#include "utils/models/Point.h"
#include "utils/math/Helpers.h"
#include "dewarping/DewarpingHelper.h"
#include "virtualcamera/DisplayImageBuilder.h"
#include "utils/images/ImageConverter.h"

VideoThread::VideoThread(std::unique_ptr<IVideoInput> videostream, std::unique_ptr<IFisheyeDewarper> dewarper,
                         std::unique_ptr<IObjectFactory> objectFactory, std::unique_ptr<IVideoOutput> imageConsumer,
                         std::unique_ptr<ISynchronizer> synchronizer, std::unique_ptr<VirtualCameraManager> virtualCameraManager,
                         std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<AngleRect>>> detectionQueue,
                         std::shared_ptr<LockTripleBuffer<Image>> imageBuffer, const DewarpingConfig& dewarpingConfig,
                         const CameraConfig& cameraConfig)
    : videoInput_(std::move(videostream))
    , dewarper_(std::move(dewarper))
    , objectFactory_(std::move(objectFactory))
    , videoOutput_(std::move(imageConsumer))
    , synchronizer_(std::move(synchronizer))
    , virtualCameraManager_(std::move(virtualCameraManager))
    , detectionQueue_(detectionQueue)
    , imageBuffer_(imageBuffer)
    , dewarpingConfig_(dewarpingConfig)
    , cameraConfig_(cameraConfig)
{
    if (!videoInput_ || !dewarper_ || !objectFactory_ || !videoOutput_ || !synchronizer_ ||
        !virtualCameraManager_ || !detectionQueue_ || !imageBuffer_)
    {
        throw std::invalid_argument("Error in VideoProcessorThread - Null is not a valid argument");
    } 
}

void VideoThread::run()
{
    try
    {
        // These are required temporarily until camera writes to device buffer
        HeapObjectFactory heapObjectFactory;
#ifndef NO_CUDA
        heapObjectFactory.allocateObjectLockTripleBuffer(*imageBuffer_);
#endif

        const Dim2<int>& resolution = cameraConfig_.resolution;
        Dim2<int> displayDim(800, 600);
        DualBuffer<Image> displayBuffers(Image(displayDim, ImageFormat::RGB_FMT));
        DisplayImageBuilder displayImageBuilder(displayDim);
        heapObjectFactory.allocateObjectDualBuffer(displayBuffers);
        displayImageBuilder.setDisplayImageColor(displayBuffers.getCurrent());
        displayImageBuilder.setDisplayImageColor(displayBuffers.getInUse());

        Dim2<int> maxVcDim = displayImageBuilder.getMaxVirtualCameraDim();
        std::vector<Image> vcImages(5, RGBImage(maxVcDim.width, maxVcDim.height));
        objectFactory_->allocateObjectVector(vcImages);

        Point<float> fisheyeCenter(resolution.width / 2.f, resolution.height / 2.f);
        std::vector<AngleRect> detections;

        // Prefetch an image

        const Image& imageCur = imageBuffer_->getCurrent();
        Image rawImage = videoInput_->readImage();

        ImageConverter imageConverter;
        imageConverter.convert(rawImage, imageCur);
        videoOutput_->writeImage(imageCur);

#ifndef NO_CUDA
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(imageCur.deviceData, imageCur.hostData, imageCur.size, cudaMemcpyHostToDevice, stream);
#endif

        // Timing variables TODO move this in a class

        int frameTimeTargetMs = 1000 / cameraConfig_.fpsTarget;
        int frameTimeDeltaMs = 0;
        int actualFrameTime = 0;
        int frameTimeModifiedTargetMs = frameTimeTargetMs;
        auto startTime = std::chrono::steady_clock::now();

        // Video loop start

        std::cout << "VideoThread loop started" << std::endl;

        while (!isAbortRequested())
        {
            if (detectionQueue_->try_dequeue(detections))
            {
                virtualCameraManager_->updateVirtualCamerasGoal(detections);
            }

            virtualCameraManager_->updateVirtualCameras(actualFrameTime);

#ifndef NO_CUDA
            cudaStreamSynchronize(stream);
#endif
            imageBuffer_->swap();
            const Image& image = imageBuffer_->getCurrent();
            rawImage = videoInput_->readImage();
            imageConverter.convert(rawImage, image);

#ifndef NO_CUDA
            cudaMemcpyAsync(image.deviceData, image.hostData, image.size, cudaMemcpyHostToDevice, stream);
#endif

            const std::vector<VirtualCamera> virtualCameras = virtualCameraManager_->getVirtualCameras();
            int vcCount = (int)virtualCameras.size();

            if (vcCount > 0)
            {
                Dim2<int> resizeDim(displayImageBuilder.getVirtualCameraDim(vcCount));
                std::vector<Image> vcResizeImages(vcCount, RGBImage(resizeDim));

                for (int i = 0; i < vcCount; ++i)
                {
                    const VirtualCamera& virtualCamera = virtualCameras[i];
                    Image& vcResizeImage = vcResizeImages[i];
                    vcResizeImage.hostData = vcImages[i].hostData;
                    vcResizeImage.deviceData = vcImages[i].deviceData;

                    DewarpingParameters vcParams = getDewarpingParametersFromAngleBoundingBox(virtualCamera, fisheyeCenter, dewarpingConfig_);
                    dewarper_->dewarpImageFiltered(image, vcResizeImage, vcParams);
                }

                const Image& displayImage = displayBuffers.getCurrent();
                displayImageBuilder.clearVirtualCamerasOnDisplayImage(displayImage);

                synchronizer_->sync();
                displayImageBuilder.createDisplayImage(vcResizeImages, displayImage);
                videoOutput_->writeImage(displayImage);
                displayBuffers.swap();
            }
            
            detections.clear();

            // Timing calculations TODO move this in a class

            auto endTime = std::chrono::steady_clock::now();
            int frameTimeMs = int(std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());
                
            if (frameTimeMs < frameTimeModifiedTargetMs)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(frameTimeModifiedTargetMs - frameTimeMs));
            }

            auto prevTime = startTime;
            startTime = std::chrono::steady_clock::now();
            actualFrameTime = int(std::chrono::duration_cast<std::chrono::milliseconds>(startTime - prevTime).count());

            frameTimeDeltaMs += (frameTimeTargetMs - actualFrameTime);

            // If the frame time delta is too big, only add a delta of 1/6 of the frame time target on this frame
            if (std::abs(frameTimeDeltaMs) < frameTimeTargetMs / 6)
            {
                frameTimeModifiedTargetMs = frameTimeTargetMs + frameTimeDeltaMs;
            }
            else
            {
                frameTimeModifiedTargetMs = frameTimeTargetMs + (frameTimeTargetMs / 6) * math::sign(frameTimeDeltaMs);
            }
        }

        objectFactory_->deallocateObjectVector(vcImages);

#ifndef NO_CUDA
            heapObjectFactory.deallocateObjectLockTripleBuffer(*imageBuffer_);
#endif
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    std::cout << "VideoThread loop finished" << std::endl;
}
#include "VideoThread.h"

#include <iostream>
#include <chrono>
#include <thread>

#include "utils/objects/HeapObjectFactory.h"
#include <cuda_runtime.h>

#include "utils/models/Point.h"
#include "utils/math/Helpers.h"
#include "dewarping/DewarpingHelper.h"

VideoThread::VideoThread(std::unique_ptr<IVideoStream> videostream, std::unique_ptr<IFisheyeDewarper> dewarper,
                         std::unique_ptr<IObjectFactory> objectFactory, std::unique_ptr<IImageConsumer> imageConsumer,
                         std::unique_ptr<ISynchronizer> synchronizer, std::unique_ptr<VirtualCameraManager> virtualCameraManager,
                         std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<AngleRect>>> detectionQueue,
                         std::shared_ptr<LockTripleBuffer<Image>> imageBuffer, const DewarpingConfig& dewarpingConfig,
                         const CameraConfig& cameraConfig)
    : videostream_(std::move(videostream))
    , dewarper_(std::move(dewarper))
    , objectFactory_(std::move(objectFactory))
    , imageConsumer_(std::move(imageConsumer))
    , synchronizer_(std::move(synchronizer))
    , virtualCameraManager_(std::move(virtualCameraManager))
    , detectionQueue_(detectionQueue)
    , imageBuffer_(imageBuffer)
    , dewarpingConfig_(dewarpingConfig)
    , cameraConfig_(cameraConfig)
{
    if (!videostream_ || !dewarper_ || !objectFactory_ || !imageConsumer_ || !synchronizer_ ||
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
#ifndef NO_CUDA
            HeapObjectFactory heapObjectFactory;
            heapObjectFactory.allocateObjectLockTripleBuffer(*imageBuffer_);
#endif

        Dim3<int> resolution;
        videostream_->getResolution(resolution);

        if (resolution != cameraConfig_.resolution)
        {
            throw std::invalid_argument("Video input resolution is not the specified one!");
        }

        std::vector<Image> vcImages(1, Image(400, 300, resolution.channels));
        objectFactory_->allocateObjectVector(vcImages);

        Point<float> fisheyeCenter(resolution.width / 2.f, resolution.height / 2.f);
        std::vector<AngleRect> detections;

        // Prefetch an image

        const Image& imageCur = imageBuffer_->getCurrent();
        videostream_->copyFrameData(imageCur);

#ifndef NO_CUDA
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(imageCur.deviceData, imageCur.hostData, imageCur.size, cudaMemcpyHostToDevice, stream);
#endif

        // Timing variables TODO move this in a class

        int64_t frameTimeTargetMs = 1000 / cameraConfig_.fpsTarget;
        int64_t frameTimeDeltaMs = 0;
        int64_t frameTimeModifiedTargetMs = frameTimeTargetMs;
        auto startTime = std::chrono::steady_clock::now();

        // Video loop start

        std::cout << "VideoThread loop started" << std::endl;

        while (!isAbortRequested())
        {
            if (detectionQueue_->try_dequeue(detections))
            {
                virtualCameraManager_->updateVirtualCamerasGoal(detections);
            }

#ifndef NO_CUDA
            cudaStreamSynchronize(stream);
#endif
            imageBuffer_->swap();
            const Image& image = imageBuffer_->getCurrent();
            videostream_->copyFrameData(image);

#ifndef NO_CUDA
            cudaMemcpyAsync(image.deviceData, image.hostData, image.size, cudaMemcpyHostToDevice, stream);
#endif

            const std::vector<VirtualCamera> virtualCameras = virtualCameraManager_->getVirtualCameras();

            for (const VirtualCamera& virtualCamera : virtualCameras)
            {
                DewarpingParameters vcParams = dewarping::getDewarpingParametersFromAngleBoundingBox(virtualCamera, fisheyeCenter, dewarpingConfig_);
                dewarper_->dewarpImageFiltered(image, vcImages[0], vcParams);

                synchronizer_->sync();

                imageConsumer_->consumeImage(vcImages[0]);
            }

            detections.clear();

            // Timing calculations TODO move this in a class

            auto endTime = std::chrono::steady_clock::now();
            int64_t frameTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
                
            if (frameTimeMs < frameTimeModifiedTargetMs)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(frameTimeModifiedTargetMs - frameTimeMs));
            }

            auto prevTime = startTime;
            startTime = std::chrono::steady_clock::now();
            int64_t actualFrameTime = std::chrono::duration_cast<std::chrono::milliseconds>(startTime - prevTime).count();

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
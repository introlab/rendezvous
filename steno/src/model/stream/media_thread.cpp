#include "media_thread.h"

#include <cstring>
#include <iostream>

#include "model/audio_suppresser/audio_suppresser.h"
#include "model/classifier/classifier.h"
#include "model/stream/audio/audio_config.h"
#include "model/stream/utils/alloc/heap_object_factory.h"
#include "model/stream/utils/images/image_drawing.h"
#include "model/stream/utils/models/circular_buffer.h"
#include "model/stream/utils/models/point.h"
#include "model/stream/video/dewarping/dewarping_helper.h"
#include "model/stream/video/dewarping/models/dewarping_config.h"
#include "model/stream/video/video_config.h"
#include "model/stream/video/video_stabilizer.h"
#include "model/stream/video/virtualcamera/display_image_builder.h"

namespace Model
{
MediaThread::MediaThread(std::unique_ptr<IAudioSource> audioSource, std::unique_ptr<IAudioSink> audioSink,
                         std::unique_ptr<IPositionSource> positionSource, std::unique_ptr<IVideoInput> videoInput,
                         std::unique_ptr<IFisheyeDewarper> dewarper, std::unique_ptr<IObjectFactory> objectFactory,
                         std::unique_ptr<IVideoOutput> videoOutput, std::unique_ptr<ISynchronizer> synchronizer,
                         std::unique_ptr<VirtualCameraManager> virtualCameraManager,
                         std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>> detectionQueue,
                         std::shared_ptr<LockTripleBuffer<Image>> imageBuffer,
                         std::unique_ptr<IImageConverter> imageConverter, std::shared_ptr<Config> config)
    : audioSource_(std::move(audioSource))
    , audioSink_(std::move(audioSink))
    , positionSource_(std::move(positionSource))
    , videoInput_(std::move(videoInput))
    , dewarper_(std::move(dewarper))
    , objectFactory_(std::move(objectFactory))
    , videoOutput_(std::move(videoOutput))
    , synchronizer_(std::move(synchronizer))
    , virtualCameraManager_(std::move(virtualCameraManager))
    , detectionQueue_(detectionQueue)
    , imageBuffer_(imageBuffer)
    , imageConverter_(std::move(imageConverter))
    , dewarpingConfig_(config->dewarpingConfig())
    , videoInputConfig_(config->videoInputConfig())
    , videoOutputConfig_(config->videoOutputConfig())
    , audioInputConfig_(config->audioInputConfig())
    , audioOutputConfig_(config->audioOutputConfig())
{
    if (!audioSource_ || !audioSink_ || !positionSource_ || !videoInput_ || !dewarper_ || !objectFactory_ ||
        !videoOutput_ || !synchronizer_ || !virtualCameraManager_ || !detectionQueue_ || !imageBuffer_ ||
        !imageConverter_)
    {
        throw std::invalid_argument("Error in MediaThread - Null is not a valid argument");
    }
}

/**
 * @brief Managing odas threads for audio and localization + camera and images processing.
 */
void MediaThread::run()
{
    // Utilitary objects
    HeapObjectFactory heapObjectFactory;
    DisplayImageBuilder displayImageBuilder(videoOutputConfig_->resolution);
    VideoStabilizer videoStabilizer(videoInputConfig_->fpsTarget);

    // Display images
    Image emptyDisplay(videoOutputConfig_->resolution, videoOutputConfig_->imageFormat);
    CircularBuffer<Image> displayBuffers(2, Image(videoOutputConfig_->resolution, videoOutputConfig_->imageFormat));

    // Virtual cameras images
    const Dim2<int>& maxVcDim = displayImageBuilder.getMaxVirtualCameraDim();
    std::vector<Image> vcImages(1, RGBImage(maxVcDim.width, maxVcDim.height));
    std::vector<Image> vcOutputFormatImages(1, Image(maxVcDim.width, maxVcDim.height, videoOutputConfig_->imageFormat));

    // TODO: config?
    const float classifierRangeThreshold = 0.26;    // ~15 degrees

    if (videoInputConfig_->fpsTarget == 0)
    {
        qCritical() << "MediaThread: target fps cannot be zero";
        return;
    }

    int chunkDurationMs = 1000 / videoInputConfig_->fpsTarget;

    try
    {
        // Allocate display images
        heapObjectFactory.allocateObject(emptyDisplay);
        heapObjectFactory.allocateObjectCircularBuffer(displayBuffers);

        // Allocate virtual camera images
        objectFactory_->allocateObjectVector(vcImages);
        objectFactory_->allocateObjectVector(vcOutputFormatImages);

        // Set background color of empty display
        displayImageBuilder.setDisplayImageColor(emptyDisplay);

        // Start audio and video resources
        audioSource_->open();
        audioSink_->open();
        positionSource_->open();
        videoInput_->open();
        videoOutput_->open();

        Point<float> fisheyeCenter(videoInputConfig_->resolution.width / 2.f,
                                   videoInputConfig_->resolution.height / 2.f);
        std::vector<SphericalAngleRect> detections;

        // Media loop start
        std::cout << "MediaThread loop started" << std::endl;

        while (!isAbortRequested())
        {
            videoStabilizer.startFrame();

            // Try to get queued detections
            if (detectionQueue_->try_dequeue(detections))
            {
                virtualCameraManager_->updateVirtualCamerasGoal(detections);
            }

            // Update the position and size of virtual cameras
            virtualCameraManager_->updateVirtualCameras(videoStabilizer.getLastFrameTimeMs());

            // Read image from video input and convert it to rgb format for dewarping
            const Image& rawImage = videoInput_->readImage();
            Image& rgbImage = imageBuffer_->getCurrent();
            imageConverter_->convert(rawImage, rgbImage);
            imageBuffer_->swap();

            // Get the active virtual cameras
            const std::vector<VirtualCamera> virtualCameras = virtualCameraManager_->getVirtualCameras();
            int vcCount = static_cast<int>(virtualCameras.size());

            // If there are active virtual cameras, dewarp images of each vc and combine them in an output image
            if (vcCount > 0)
            {
                // Dynamically allocate more virtual camera images
                for (int i = vcImages.size(); i < vcCount; ++i)
                {
                    RGBImage vcImage(maxVcDim.width, maxVcDim.height);
                    objectFactory_->allocateObject(vcImage);
                    vcImages.push_back(vcImage);

                    Image vcOutputFormatImage(maxVcDim.width, maxVcDim.height, videoOutputConfig_->imageFormat);
                    objectFactory_->allocateObject(vcOutputFormatImage);
                    vcOutputFormatImages.push_back(vcOutputFormatImage);
                }

                // Get the size of the virtual camera images to dewarp (this is to prevent resize in the output format)
                Dim2<int> resizeDim(displayImageBuilder.getVirtualCameraDim(vcCount));
                std::vector<Image> vcResizeImages(vcCount, RGBImage(resizeDim));
                std::vector<Image> vcResizeOutputFormatImages(vcCount,
                                                              Image(resizeDim, videoOutputConfig_->imageFormat));

                // Virtual camera dewarping loop
                for (int i = 0; i < vcCount; ++i)
                {
                    // Use the same buffers as vcImages for the smaller dewarped images
                    Image& vcResizeImage = vcResizeImages[i];
                    vcResizeImage.hostData = vcImages[i].hostData;
                    vcResizeImage.deviceData = vcImages[i].deviceData;

                    // Dewarping of virtual camera
                    const VirtualCamera& virtualCamera = virtualCameras[i];
                    DewarpingParameters vcParams =
                        getDewarpingParametersFromAngleBoundingBox(virtualCamera, fisheyeCenter, dewarpingConfig_);
                    dewarper_->dewarpImageFiltered(rgbImage, vcResizeImage, vcParams);

                    // Use the same buffers as vcOutputFormatImages for the smaller dewarped (and converted) images
                    Image& vcResizeOutputFormatImage = vcResizeOutputFormatImages[i];
                    vcResizeOutputFormatImage.hostData = vcOutputFormatImages[i].hostData;
                    vcResizeOutputFormatImage.deviceData = vcOutputFormatImages[i].deviceData;

                    // Conversion from rgb to output format
                    imageConverter_->convert(vcResizeImage, vcResizeOutputFormatImage);
                }

                // Clear the image before writting to it
                Image& displayImage = displayBuffers.current();
                std::memcpy(displayImage.hostData, emptyDisplay.hostData, displayImage.size);

                // Set the timestamp of the output image to the timestamp of the input image
                displayImage.timeStamp = rawImage.timeStamp;

                // Wait for dewarping to be completed
                synchronizer_->sync();

                // Get audio sources and image spatial positions
                std::vector<SourcePosition> sourcePositions = positionSource_->getPositions();
                std::vector<SphericalAngleRect> imagePositions;
                imagePositions.reserve(virtualCameras.size());
                for (const auto& vc : virtualCameras)
                {
                    imagePositions.push_back(vc);
                }

                int borderWidth = 2;
                RGB borderColor;
                borderColor.r = 0;
                borderColor.g = 165;
                borderColor.b = 89;

                std::vector<std::pair<int, int>> audioImagePairs =
                    Classifier::getAudioImagePairs(sourcePositions, imagePositions, classifierRangeThreshold);
                // temp
                // std::pair<int, int> pair(0, 0);
                // audioImagePairs.push_back(pair);
                // ---

                for (std::pair<int, int> pair : audioImagePairs)
                {
                    ImageDrawing::drawBordersUYVY(vcResizeOutputFormatImages[pair.second], borderWidth, borderColor);
                }

                // Write to output image and send it to the video output
                displayImageBuilder.createDisplayImage(vcResizeOutputFormatImages, displayImage);
                videoOutput_->writeImage(displayImage);
                displayBuffers.next();

                int readCount = videoStabilizer.getLastFrameTimeMs() / (chunkDurationMs / 2) + 1;

                // Remove unwanted audio sources and write audio into audio sink
                AudioChunk audioChunk;
                while (audioSource_->readAudioChunk(audioChunk) && readCount > 0)
                {
                    std::vector<int> sourcesToKeep =
                        Classifier::getSourcesToKeep(sourcePositions, imagePositions, classifierRangeThreshold);

                    AudioSuppresser::suppressNoise(sourcesToKeep, audioChunk);

                    audioSink_->write(audioChunk.audioData.get(), audioChunk.size);

                    --readCount;
                }
            }
            else
            {
                // If there are no active virtual cameras, just send an empty image
                videoOutput_->writeImage(emptyDisplay);
            }

            detections.clear();

            // If the frame took less than 1/fps, this call will block to match frame time of 1/fps
            videoStabilizer.endFrame();
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "Error in media thread : " << e.what() << std::endl;
    }

    // Clean audio and video resources
    audioSource_->close();
    audioSink_->close();
    positionSource_->close();
    videoInput_->close();
    videoOutput_->close();
    virtualCameraManager_->clearVirtualCameras();

    // Deallocate display images
    heapObjectFactory.deallocateObject(emptyDisplay);
    heapObjectFactory.deallocateObjectCircularBuffer(displayBuffers);

    // Deallocate virtual camera images
    objectFactory_->deallocateObjectVector(vcImages);
    objectFactory_->deallocateObjectVector(vcOutputFormatImages);

    std::cout << "MediaThread loop finished" << std::endl;
}
}    // namespace Model

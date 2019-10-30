#include "media_thread.h"

#include <iostream>
#include <cstring>

#include "model/audio_suppresser/audio_suppresser.h"
#include "model/classifier/classifier.h"
#include "model/stream/utils/alloc/heap_object_factory.h"
#include "model/stream/utils/models/point.h"
#include "model/stream/video/dewarping/dewarping_helper.h"
#include "model/stream/video/video_stabilizer.h"
#include "model/stream/video/virtualcamera/display_image_builder.h"
#include "model/stream/utils/models/circular_buffer.h"

namespace Model
{
MediaThread::MediaThread(std::unique_ptr<IAudioSource> audioSource, std::unique_ptr<IAudioSink> audioSink,
                         std::unique_ptr<IPositionSource> positionSource, std::unique_ptr<IVideoInput> videoInput,
                         std::unique_ptr<IFisheyeDewarper> dewarper, std::unique_ptr<IObjectFactory> objectFactory,
                         std::unique_ptr<IVideoOutput> videoOutput, std::unique_ptr<ISynchronizer> synchronizer,
                         std::unique_ptr<VirtualCameraManager> virtualCameraManager,
                         std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>> detectionQueue,
                         std::shared_ptr<LockTripleBuffer<Image>> imageBuffer,
                         std::unique_ptr<IImageConverter> imageConverter, const DewarpingConfig& dewarpingConfig,
                         const VideoConfig& videoInputConfig, const VideoConfig& videoOutputConfig,
                         const AudioConfig& audioInputConfig, const AudioConfig& audioOutputConfig)
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
    , dewarpingConfig_(dewarpingConfig)
    , videoInputConfig_(videoInputConfig)
    , videoOutputConfig_(videoOutputConfig)
    , audioInputConfig_(audioInputConfig)
    , audioOutputConfig_(audioOutputConfig)
{
    if (!audioSource_ || !audioSink_ || !positionSource_ || !videoInput_ || !dewarper_ || !objectFactory_ ||
        !videoOutput_ || !synchronizer_ || !virtualCameraManager_ || !detectionQueue_ || !imageBuffer_ ||
        !imageConverter_)
    {
        throw std::invalid_argument("Error in MediaThread - Null is not a valid argument");
    }
}

void MediaThread::run()
{
    // Utilitary objects
    HeapObjectFactory heapObjectFactory;
    DisplayImageBuilder displayImageBuilder(videoOutputConfig_.resolution);
    VideoStabilizer videoStabilizer(videoInputConfig_.fpsTarget);

    // Display images
    Image emptyDisplay(videoOutputConfig_.resolution, videoOutputConfig_.imageFormat);
    CircularBuffer<Image> displayBuffers(2, Image(videoOutputConfig_.resolution, videoOutputConfig_.imageFormat));

    // Virtual cameras images
    Dim2<int> maxVcDim = displayImageBuilder.getMaxVirtualCameraDim();
    std::vector<Image> vcImages(1, RGBImage(maxVcDim.width, maxVcDim.height));
    std::vector<Image> vcOutputFormatImages(1, Image(maxVcDim.width, maxVcDim.height, videoOutputConfig_.imageFormat));
    
    // TODO: config?
    const int classifierRangeThreshold = 2;

    // TODO: will be managed by odas audio source
    uint8_t* audioBuffer = new uint8_t[audioInputConfig_.bufferSize];

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

        // Start audio resources
        audioSource_->open();
        audioSink_->open();
        positionSource_->open();

        Point<float> fisheyeCenter(videoInputConfig_.resolution.width / 2.f, videoInputConfig_.resolution.height / 2.f);
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

            // Read audio source and positions
            int audioBytesRead = audioSource_->read(audioBuffer, sizeof(audioBuffer));
            std::vector<SourcePosition> sourcePositions = positionSource_->getPositions();

            // Read image from video input and convert it to rgb format for dewarping
            const Image& rawImage = videoInput_->readImage();
            const Image& rgbImage = imageBuffer_->getCurrent();
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

                    Image vcOutputFormatImage(maxVcDim.width, maxVcDim.height, videoOutputConfig_.imageFormat);
                    objectFactory_->allocateObject(vcOutputFormatImage);
                    vcOutputFormatImages.push_back(vcOutputFormatImage);
                }

                // Get the size of the virtual camera images to dewarp (this is to prevent resize in the output format)
                Dim2<int> resizeDim(displayImageBuilder.getVirtualCameraDim(vcCount));
                std::vector<Image> vcResizeImages(vcCount, RGBImage(resizeDim));
                std::vector<Image> vcResizeOutputFormatImages(vcCount, Image(resizeDim, videoOutputConfig_.imageFormat));

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
                    Image& vcResizeOutputFormatImage =  vcResizeOutputFormatImages[i];
                    vcResizeOutputFormatImage.hostData = vcOutputFormatImages[i].hostData;
                    vcResizeOutputFormatImage.deviceData = vcOutputFormatImages[i].deviceData;

                    // Conversion from rgb to output format
                    imageConverter_->convert(vcResizeImage, vcResizeOutputFormatImage);
                }

                // Clear the image before writting to it
                const Image& displayImage = displayBuffers.current();
                std::memcpy(displayImage.hostData, emptyDisplay.hostData, displayImage.size);

                // Wait for dewarping to be completed
                synchronizer_->sync();

                // Write to output image and send it to the video output
                displayImageBuilder.createDisplayImage(vcResizeOutputFormatImages, displayImage);
                videoOutput_->writeImage(displayImage);
                displayBuffers.next();
            }
            else
            {
                // If there are no active virtual cameras, just send an empty image
                videoOutput_->writeImage(emptyDisplay);
            }

            if (audioBytesRead > 0)
            {
                std::vector<SphericalAngleRect> imagePositions;
                imagePositions.reserve(virtualCameras.size());
                for (auto vc : virtualCameras)
                {
                    imagePositions.push_back(vc);
                }

                std::vector<int> sourcesToSuppress =
                    Classifier::classify(sourcePositions, imagePositions, classifierRangeThreshold);

                AudioSuppresser::suppressSources(sourcesToSuppress, audioBuffer, audioBytesRead);

                audioSink_->write(audioBuffer, audioBytesRead);
            }

            detections.clear();

            // If the frame took less than 1/fps, this call will block to match frame time of 1/fps
            videoStabilizer.endFrame();
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "Error in video thread : " << e.what() << std::endl;
    }

    // Clean audio resources
    audioSource_->close();
    audioSink_->close();
    positionSource_->close();
    delete[] audioBuffer;
   
    // Deallocate display images
    heapObjectFactory.allocateObject(emptyDisplay);
    heapObjectFactory.deallocateObjectCircularBuffer(displayBuffers);

    // Deallocate virtual camera images
    objectFactory_->deallocateObjectVector(vcImages);
    objectFactory_->deallocateObjectVector(vcOutputFormatImages);

    std::cout << "MediaThread loop finished" << std::endl;
}
}    // namespace Model

#include "dewarped_video_input.h"

#include <cstring>
#include <iostream>

#include "model/classifier/classifier.h"
#include "model/stream/utils/alloc/heap_object_factory.h"
#include "model/stream/utils/images/image_drawing.h"
#include "model/stream/video/virtualcamera/display_image_builder.h"
#include "model/stream/frame_rate_stabilizer.h"
#include "model/stream/utils/models/point.h"
#include "model/stream/utils/models/spherical_angle_rect.h"
#include "model/stream/video/dewarping/dewarping_helper.h"

namespace Model
{
DewarpedVideoInput::DewarpedVideoInput(std::unique_ptr<IVideoInput> videoInput,
                                       std::unique_ptr<IFisheyeDewarper> dewarper, std::unique_ptr<IObjectFactory> objectFactory,
                                       std::unique_ptr<ISynchronizer> synchronizer,
                                       std::shared_ptr<VirtualCameraManager> virtualCameraManager,
                                       std::unique_ptr<DetectionThread> detectionThread,
                                       std::shared_ptr<LockTripleBuffer<RGBImage>> imageBuffer, std::unique_ptr<IImageConverter> imageConverter,
                                       std::shared_ptr<IPositionSource> positionSource,
                                       std::shared_ptr<DewarpingConfig> dewarpingConfig, std::shared_ptr<VideoConfig> videoInputConfig,
                                       std::shared_ptr<VideoConfig> videoOutputConfig,
                                       int bufferCount,
                                       float classifierRangeThreshold)
    : Thread()
    , videoInput_(std::move(videoInput))
    , dewarper_(std::move(dewarper))
    , objectFactory_(std::move(objectFactory))
    , synchronizer_(std::move(synchronizer))
    , virtualCameraManager_(virtualCameraManager)
    , detectionThread_(std::move(detectionThread))
    , imageBuffer_(imageBuffer)
    , imageConverter_(std::move(imageConverter))
    , positionSource_(positionSource)
    , dewarpingConfig_(dewarpingConfig)
    , videoInputConfig_(videoInputConfig)
    , videoOutputConfig_(videoOutputConfig)
    , outputImageQueue_(bufferCount - 1)
    , bufferCount_(bufferCount)
    , classifierRangeThreshold_(classifierRangeThreshold)
{
    if (!videoInput_ || !dewarper_ || !objectFactory_ || !synchronizer_ || !virtualCameraManager_ || 
        !imageBuffer_ || !imageConverter_ || !positionSource || !dewarpingConfig_ || !videoInputConfig_ || !videoOutputConfig_)
    {
        throw std::invalid_argument("Error in DewarpedVideoInput - Null is not a valid argument");
    }

    if (bufferCount < 2)
    {
        throw std::invalid_argument("Error in DewarpedVideoInput - need at least a buffer size of 2");
    }

    fisheyeCenter_ = Point<float>(videoInputConfig_->resolution.width / 2.f, 
                                  videoInputConfig_->resolution.height / 2.f);
}

void DewarpedVideoInput::open()
{
    detectionThread_->start();
    start();
}

void DewarpedVideoInput::close()
{
    if (detectionThread_->getState() != Thread::ThreadStatus::CRASHED)
    {
        detectionThread_->stop();
        detectionThread_->join();
    }

    stop();
    join();
}

bool DewarpedVideoInput::readImage(Image& image)
{
    return outputImageQueue_.try_dequeue(image);
}

void DewarpedVideoInput::run()
{
    // Utilitary objects
    HeapObjectFactory heapObjectFactory;
    DisplayImageBuilder displayImageBuilder(videoOutputConfig_->resolution);
    FrameRateStabilizer videoStabilizer(videoInputConfig_->fpsTarget);

    // Display images
    Image emptyDisplay(videoOutputConfig_->resolution, videoOutputConfig_->imageFormat);
    CircularBuffer<Image> displayBuffers(bufferCount_, Image(videoOutputConfig_->resolution, videoOutputConfig_->imageFormat));

    // Virtual cameras images
    Dim2<int> maxVcDim = displayImageBuilder.getMaxVirtualCameraDim();

    try
    {
        // Allocate display images
        heapObjectFactory.allocateObject(emptyDisplay);
        heapObjectFactory.allocateObjectCircularBuffer(displayBuffers);

        // Set background color of empty display
        displayImageBuilder.setDisplayImageColor(emptyDisplay);

        // Start video resources
        videoInput_->open();

        while (!isAbortRequested())
        {
            videoStabilizer.startFrame();

            updateVirtualCameras(videoStabilizer.getLastFrameTimeMs());

            // Read image from video input and convert it to rgb format for dewarping
            const RGBImage& rgbFisheyeImage = getRgbFisheyeImage();

            // Get the active virtual cameras
            const std::vector<VirtualCamera> virtualCameras = virtualCameraManager_->getVirtualCameras();
            int vcCount = static_cast<int>(virtualCameras.size());

            // If there are active virtual cameras, dewarp images of each vc and combine them in an output image
            if (vcCount > 0)
            {
                // Dynamically allocate more virtual camera image buffers
                for (int i = vcRgbImages_.size(); i < vcCount; ++i)
                {
                    addDewarpedImageBuffers(maxVcDim);
                }

                // Get the size of the virtual camera images to dewarp (this is to prevent resizing after)
                Dim2<int> dewarpDim = displayImageBuilder.getVirtualCameraDim(vcCount);
                std::vector<Image> dewarpedImages(vcCount);

                // Virtual camera dewarping loop
                for (int i = 0; i < vcCount; ++i)
                {
                    dewarpedImages[i] = dewarpInOutputFormat(rgbFisheyeImage, virtualCameras[i], dewarpDim, 
                                                             vcRgbImages_[i], vcOutputFormatImages_[i]);
                }

                // Clear the image before writting to it
                Image& displayImage = displayBuffers.current();
                std::memcpy(displayImage.hostData, emptyDisplay.hostData, displayImage.size);

                // Set the timestamp of the output image to the timestamp of the input image
                displayImage.timeStamp = rgbFisheyeImage.timeStamp;

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
                    Classifier::getAudioImagePairs(sourcePositions, imagePositions, classifierRangeThreshold_);

                for (std::pair<int, int> pair : audioImagePairs)
                {
                    ImageDrawing::drawBorders(dewarpedImages[pair.second], ImageFormat::UYVY_FMT,
                                              borderWidth, borderColor);
                }

                // Write to output image and send it to the video output
                displayImageBuilder.createDisplayImage(dewarpedImages, displayImage);
                queueOutputImage(displayImage);
                displayBuffers.next();
            }
            else
            {
                // Set the timestamp of the output image to the timestamp of the input image
                emptyDisplay.timeStamp = rgbFisheyeImage.timeStamp;

                // If there are no active virtual cameras, just send an empty image
                queueOutputImage(emptyDisplay);
            }

            // If the frame took less than 1/fps, this call will block to match frame time of 1/fps
            videoStabilizer.endFrame();
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "Error in DewarpedVideoInput thread : " << e.what() << std::endl;
    }

    // Clean video resources
    videoInput_->close();
    virtualCameraManager_->clearVirtualCameras();

    // Deallocate display images
    heapObjectFactory.deallocateObject(emptyDisplay);
    heapObjectFactory.deallocateObjectCircularBuffer(displayBuffers);

    cleanDewarpedImageBuffers();

    std::cout << "DewarpedVideoInput loop finished" << std::endl;
}

void DewarpedVideoInput::queueOutputImage(const Image& image)
{
    bool success = false;
    
    // If queue is full keep trying...
    while (!success && !isAbortRequested())
    {
        success = outputImageQueue_.try_enqueue(image);

        if (!success)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void DewarpedVideoInput::updateVirtualCameras(int frameTimeMs)
{
    std::vector<SphericalAngleRect> detections;
    
    // Try to get queued detections
    if (detectionThread_->getDetections(detections))
    {
        virtualCameraManager_->updateVirtualCamerasGoal(detections);
    }

    // Update the position and size of virtual cameras
    virtualCameraManager_->updateVirtualCameras(frameTimeMs);
}

const RGBImage& DewarpedVideoInput::getRgbFisheyeImage()
{
    Image rawFisheyeImage;
    videoInput_->readImage(rawFisheyeImage);

    // Convert the fisheye image to rgb format
    RGBImage& rgbFisheyeImage = imageBuffer_->getCurrent();
    imageConverter_->convert(rawFisheyeImage, rgbFisheyeImage);

    // Change the buffer returned by getCurrent()
    imageBuffer_->swap();

    return rgbFisheyeImage;
}

Image DewarpedVideoInput::dewarpInOutputFormat(const RGBImage& rgbFisheyeImage, const SphericalAngleRect& dewarpArea,
                                               const Dim2<int>& dewarpDim, const RGBImage& allocatedRgbImage, 
                                               const Image& allocatedOutputImage)
{
    RGBImage dewarpedRgbImage(dewarpDim);
    Image dewarpedOutputImage(dewarpDim, allocatedOutputImage.format);

    // Use the allocated rgb buffer, but with correct dewarp dimension
    dewarpedRgbImage.hostData = allocatedRgbImage.hostData;
    dewarpedRgbImage.deviceData = allocatedRgbImage.deviceData;

    // Dewarping of virtual camera
    DewarpingParameters vcParams = getDewarpingParametersFromSphericalAngleRect(dewarpArea, *dewarpingConfig_, fisheyeCenter_);
    dewarper_->dewarpImageFiltered(rgbFisheyeImage, dewarpedRgbImage, vcParams);

    // Use the allocated buffer, but with correct dewarp dimension
    dewarpedOutputImage.hostData = allocatedOutputImage.hostData;
    dewarpedOutputImage.deviceData = allocatedOutputImage.deviceData;

    // Conversion from rgb to output format
    imageConverter_->convert(dewarpedRgbImage, dewarpedOutputImage);

    return dewarpedOutputImage;
}

void DewarpedVideoInput::addDewarpedImageBuffers(const Dim2<int> maxVcDim)
{
    RGBImage vcImage(maxVcDim.width, maxVcDim.height);
    objectFactory_->allocateObject(vcImage);
    vcRgbImages_.push_back(vcImage);

    Image vcOutputFormatImage(maxVcDim.width, maxVcDim.height, videoOutputConfig_->imageFormat);
    objectFactory_->allocateObject(vcOutputFormatImage);
    vcOutputFormatImages_.push_back(vcOutputFormatImage);
}

void DewarpedVideoInput::cleanDewarpedImageBuffers()
{
    // Deallocate virtual camera images
    objectFactory_->deallocateObjectVector(vcRgbImages_);
    objectFactory_->deallocateObjectVector(vcOutputFormatImages_);

    // Clear virtual camera images
    vcRgbImages_.clear();
    vcOutputFormatImages_.clear();
}

}   // namespace Model

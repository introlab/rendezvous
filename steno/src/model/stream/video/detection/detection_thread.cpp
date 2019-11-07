#include "detection_thread.h"

#include <iostream>

#include "model/stream/utils/math/math_constants.h"
#include "model/stream/video/dewarping/dewarping_helper.h"
#include "model/stream/video/input/image_file_reader.h"

namespace Model
{
DetectionThread::DetectionThread(std::shared_ptr<LockTripleBuffer<Image>> imageBuffer, std::unique_ptr<IDetector> detector,
    std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>> detectionQueue,
    std::unique_ptr<IDetectionFisheyeDewarper> dewarper, std::unique_ptr<IObjectFactory> objectFactory,
    std::unique_ptr<ISynchronizer> synchronizer, std::shared_ptr<DewarpingConfig> dewarpingConfig)
    : Thread()
    , imageBuffer_(imageBuffer)
    , detector_(std::move(detector))
    , dewarper_(std::move(dewarper))
    , objectFactory_(std::move(objectFactory))
    , synchronizer_(std::move(synchronizer))
    , detectionQueue_(detectionQueue)
    , dewarpingConfig_(dewarpingConfig)
    , dewarpCount_(dewarpingConfig->value(DewarpingConfig::DETECTION_DEWARPING_COUNT).toInt())
{
    if (!imageBuffer_ || !detector_ || !dewarper_ || !objectFactory_ || !synchronizer_ || !detectionQueue_)
    {
        throw std::invalid_argument("Error in DetectionThread - Arguments can not be null");
    }
}

void DetectionThread::run()
{
    Dim2<int> resolution(imageBuffer_->getCurrent());
    Dim2<int> detectionResolution(detector_->getInputImageDim());
    Point<float> fisheyeCenter(resolution.width / 2.f, resolution.height / 2.f);

    std::vector<DewarpingParameters> dewarpingParams;
    std::vector<ImageFloat> detectionImages;
    std::vector<DewarpingMapping> dewarpingMappings;
    std::vector<SphericalAngleRect> detections;

    try
    {
        // Allocate and prepare objects for dewarping and detection
        dewarpingParams = getDetectionDewarpingParameters(resolution, dewarpCount_);
        detectionImages = getDetectionImages(detectionResolution, dewarpCount_);
        dewarpingMappings = getDewarpingMappings(dewarpingParams, resolution, detectionResolution, dewarpCount_);

        std::cout << "DetectionThread loop started" << std::endl;

        while (!isAbortRequested())
        {
            // Make sure a new image is actually in the buffer
            while (!imageBuffer_->getAndClearSwapCount() && !isAbortRequested())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            // No need to do a detection if thread is aborted
            if (isAbortRequested())
            {
                break;
            }

            // We lock the image so we can use it without it being overwritten
            imageBuffer_->lockInUse();
            const Image& image = imageBuffer_->getLocked();

            // Dewarp each view, detect on each view and concatenate the results
            for (int i = 0; i < dewarpCount_; ++i)
            {
                dewarper_->dewarpImage(image, detectionImages[i], dewarpingMappings[i]);
                synchronizer_->sync();
                const std::vector<Rectangle> viewDetections = detector_->detectInImage(detectionImages[i]);

                for (const Rectangle& detection : viewDetections)
                {
                    detections.push_back(getAngleRectFromDewarpedImageRectangle(detection, dewarpingParams[i],
                                                                                detectionImages[i], fisheyeCenter,
                                                                                dewarpingConfig_->fisheyeAngle));
                }
            }

            bool success = false;

            // Output the detections, if queue is full keep trying...
            while (!success && !isAbortRequested())
            {
                success = detectionQueue_->try_enqueue(std::move(detections));

                if (!success)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }

            detections.clear();    // Just making sure... Should be empty because of the std::move
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "Error in detection thread : " << e.what() << std::endl;
    }

    objectFactory_->deallocateObjectVector(detectionImages);
    objectFactory_->deallocateObjectVector(dewarpingMappings);

    std::cout << "DetectionThread loop finished" << std::endl;
}

std::vector<DewarpingParameters> DetectionThread::getDetectionDewarpingParameters(const Dim2<int>& dim, int dewarpCount)
{
    std::vector<DewarpingParameters> dewarpingParams;
    dewarpingParams.reserve(dewarpCount);

    float angleSpan = (2.f * math::PI) / dewarpCount;

    for (int i = 0; i < dewarpCount; ++i)
    {
        dewarpingParams.push_back(getDewarpingParameters(dim, dewarpingConfig_, i * angleSpan));
    }

    return dewarpingParams;
}

std::vector<ImageFloat> DetectionThread::getDetectionImages(const Dim2<int>& dim, int dewarpCount)
{
    std::vector<ImageFloat> detectionImages;
    detectionImages.resize(dewarpCount, RGBImageFloat(dim));

    objectFactory_->allocateObjectVector(detectionImages);

    for (ImageFloat& image : detectionImages)
    {
        dewarper_->prepareOutputImage(image);
    }

    synchronizer_->sync();    // Wait for images to be prepared

    return detectionImages;
}

std::vector<DewarpingMapping> DetectionThread::getDewarpingMappings(
    const std::vector<DewarpingParameters>& dewarpingParams, const Dim2<int>& src, const Dim2<int>& dst,
    int dewarpCount)
{
    std::vector<DewarpingMapping> dewarpingMappings;
    Dim2<int> rectifiedDim = dewarper_->getRectifiedOutputDim(dst);
    dewarpingMappings.resize(dewarpCount, DewarpingMapping(rectifiedDim));

    objectFactory_->allocateObjectVector(dewarpingMappings);

    for (int i = 0; i < dewarpCount; ++i)
    {
        dewarper_->fillDewarpingMapping(src, dewarpingParams[i], dewarpingMappings[i]);
    }

    synchronizer_->sync();    // Wait for all the mappings to be filled

    return dewarpingMappings;
}
}    // namespace Model

#include "detection_thread.h"

#include <iostream>

#include "model/stream/utils/math/math_constants.h"
#include "model/stream/utils/math/geometry_utils.h"
#include "model/stream/video/dewarping/dewarping_helper.h"

namespace Model
{
DetectionThread::DetectionThread(
    std::shared_ptr<LockTripleBuffer<Image>> imageBuffer, std::unique_ptr<IDetector> detector,
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
{
    if (!imageBuffer_ || !detector_ || !dewarper_ || !objectFactory_ || !synchronizer_ || !detectionQueue_)
    {
        throw std::invalid_argument("Error in DetectionThread - Arguments can not be null");
    }
}

void DetectionThread::run()
{
    m_state = ThreadStatus::RUNNING;
    notify();

    Dim2<int> resolution(imageBuffer_->getCurrent());
    Dim2<int> detectionResolution(detector_->getInputImageDim());
    Dim2<int> rectifiedResolution(dewarper_->getRectifiedOutputDim(detectionResolution));
    Point<float> fisheyeCenter(resolution.width / 2.f, resolution.height / 2.f);
    RGBImageFloat detectionImage(rectifiedResolution);

    std::vector<DewarpingParameters> dewarpingParams;
    std::vector<ImageFloat> detectionImages;
    std::vector<DewarpingMapping> dewarpingMappings;
    std::vector<SphericalAngleRect> detections;

    try
    {
        // Allocate and prepare objects for dewarping and detection
        dewarpingParams = getDetectionDewarpingParameters(resolution, dewarpingConfig_->detectionDewarpingCount);
        detectionImages = getDetectionImages(detectionResolution, dewarpingConfig_->detectionDewarpingCount);
        dewarpingMappings = getDewarpingMappings(dewarpingParams, resolution, rectifiedResolution, dewarpingConfig_->detectionDewarpingCount);

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
            for (int i = 0; i < dewarpingConfig_->detectionDewarpingCount; ++i)
            {
                dewarper_->dewarpImage(image, detectionImages[i], dewarpingMappings[i]);
                synchronizer_->sync();

                // Detection image has the exact dewarped size (size of data ignoring the possible formatting required by the detector)
                detectionImage.hostData = detectionImages[i].hostData;
                detectionImage.deviceData = detectionImages[i].deviceData;

                std::vector<Rectangle> viewDetections = detector_->detectInImage(detectionImage);

                for (const Rectangle& detection : viewDetections)
                {
                    float middleAngleDiff = (2.f * math::PI) / dewarpingConfig_->detectionDewarpingCount;
                    
                    SphericalAngleRect sphericalAngleRect = getAngleRectFromDewarpedImageRectangle(detection, dewarpingParams[i],
                                                                                                   detectionImage, fisheyeCenter,
                                                                                                   dewarpingConfig_->fisheyeAngle);

                    if (!isInOverlappingZone(sphericalAngleRect, dewarpingConfig_->angleSpan, middleAngleDiff * i))
                    {
                        detections.push_back(sphericalAngleRect);
                    }
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
        m_state = ThreadStatus::CRASHED;
    }

    objectFactory_->deallocateObjectVector(detectionImages);
    objectFactory_->deallocateObjectVector(dewarpingMappings);

    std::cout << "DetectionThread loop finished" << std::endl;
    if (m_state != ThreadStatus::CRASHED)
    {
        m_state = ThreadStatus::STOPPED;
    }
    notify();
}

std::vector<DewarpingParameters> DetectionThread::getDetectionDewarpingParameters(const Dim2<int>& dim, int detectionDewarpingCount)
{
    std::vector<DewarpingParameters> dewarpingParams;
    dewarpingParams.reserve(detectionDewarpingCount);

    float middleAngleDiff = (2.f * math::PI) / detectionDewarpingCount;

    for (int i = 0; i < detectionDewarpingCount; ++i)
    {
        DonutSlice donutSlice(static_cast<float>(dim.width) / 2.f, static_cast<float>(dim.height) / 2.f, 
                              dewarpingConfig_->inRadius, dewarpingConfig_->outRadius, i * middleAngleDiff,
                              dewarpingConfig_->angleSpan);
        dewarpingParams.push_back(getDewarpingParameters(donutSlice, dewarpingConfig_->topDistorsionFactor, 
                                                         dewarpingConfig_->bottomDistorsionFactor));
    }

    return dewarpingParams;
}

std::vector<ImageFloat> DetectionThread::getDetectionImages(const Dim2<int>& dim, int detectionDewarpingCount)
{
    std::vector<ImageFloat> detectionImages;
    detectionImages.resize(detectionDewarpingCount, RGBImageFloat(dim));

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
    int detectionDewarpingCount)
{
    std::vector<DewarpingMapping> dewarpingMappings;
    dewarpingMappings.resize(detectionDewarpingCount, DewarpingMapping(dst));

    objectFactory_->allocateObjectVector(dewarpingMappings);

    for (int i = 0; i < detectionDewarpingCount; ++i)
    {
        dewarper_->fillDewarpingMapping(src, dewarpingParams[i], dewarpingMappings[i]);
    }

    synchronizer_->sync();    // Wait for all the mappings to be filled

    return dewarpingMappings;
}
}    // namespace Model

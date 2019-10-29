#include "media_thread.h"

#include <iostream>

#include "model/audio_suppresser/audio_suppresser.h"
#include "model/classifier/classifier.h"
#include "model/stream/utils/alloc/heap_object_factory.h"
#include "model/stream/utils/models/point.h"
#include "model/stream/video/dewarping/dewarping_helper.h"
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
    // TODO: config?
    const int classifierRangeThreshold = 2;

    // TODO: will be managed by odas audio source
    uint8_t* audioBuffer = new uint8_t[audioInputConfig_.bufferSize];

    try
    {
        audioSource_->open();
        audioSink_->open();
        positionSource_->open();

        HeapObjectFactory heapObjectFactory;
        DualBuffer<Image> displayBuffers(Image(videoOutputConfig_.resolution, videoOutputConfig_.imageFormat));
        DisplayImageBuilder displayImageBuilder(videoOutputConfig_.resolution);

        heapObjectFactory.allocateObjectDualBuffer(displayBuffers);
        displayImageBuilder.setDisplayImageColor(displayBuffers.getCurrent());
        displayImageBuilder.setDisplayImageColor(displayBuffers.getInUse());

        Dim2<int> maxVcDim = displayImageBuilder.getMaxVirtualCameraDim();
        std::vector<Image> vcImages(5, RGBImage(maxVcDim.width, maxVcDim.height));
        objectFactory_->allocateObjectVector(vcImages);

        Point<float> fisheyeCenter(videoInputConfig_.resolution.width / 2.f, videoInputConfig_.resolution.height / 2.f);
        std::vector<SphericalAngleRect> detections;

        VideoStabilizer videoStabilizer(videoInputConfig_.fpsTarget);

        // Media loop start

        std::cout << "MediaThread loop started" << std::endl;

        while (!isAbortRequested())
        {
            videoStabilizer.startFrame();

            if (detectionQueue_->try_dequeue(detections))
            {
                virtualCameraManager_->updateVirtualCamerasGoal(detections);
            }

            virtualCameraManager_->updateVirtualCameras(videoStabilizer.getLastFrameTimeMs());

            int audioBytesRead = audioSource_->read(audioBuffer, sizeof(audioBuffer));
            std::vector<SourcePosition> sourcePositions = positionSource_->getPositions();

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

                    DewarpingParameters vcParams =
                        getDewarpingParametersFromAngleBoundingBox(virtualCamera, fisheyeCenter, dewarpingConfig_);
                    dewarper_->dewarpImageFiltered(rgbImage, vcResizeImage, vcParams);

                    // Ok, this is a hack until we work with raw data for dewarping of virtual cameras (convert on
                    // itself, only work from RGB)
                    Image inImage = vcResizeImage;
                    vcResizeImages[i] = Image(inImage.width, inImage.height, videoOutputConfig_.imageFormat);
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

            videoStabilizer.endFrame();
        }

        heapObjectFactory.deallocateObjectDualBuffer(displayBuffers);
        objectFactory_->deallocateObjectVector(vcImages);
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    audioSource_->close();
    audioSink_->close();
    positionSource_->close();

    delete[] audioBuffer;

    std::cout << "MediaThread loop finished" << std::endl;
}
}    // namespace Model

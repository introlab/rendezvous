#include "media_thread.h"

#include <cstring>
#include <iostream>

#include "model/audio_suppresser/audio_suppresser.h"
#include "model/classifier/classifier.h"
#include "model/stream/audio/audio_config.h"
#include "model/stream/frame_rate_stabilizer.h"
#include "model/stream/utils/alloc/heap_object_factory.h"
#include "model/stream/utils/images/image_drawing.h"
#include "model/stream/utils/models/circular_buffer.h"
#include "model/stream/utils/models/point.h"
#include "model/stream/video/dewarping/dewarping_helper.h"
#include "model/stream/video/dewarping/models/dewarping_config.h"
#include "model/stream/video/video_config.h"
#include "model/stream/video/virtualcamera/display_image_builder.h"

namespace Model
{
MediaThread::MediaThread(std::unique_ptr<IAudioSource> audioSource, std::unique_ptr<IAudioSink> audioSink,
                         std::shared_ptr<IPositionSource> positionSource, std::unique_ptr<IVideoInput> videoInput,
                         std::unique_ptr<IVideoOutput> videoOutput,
                         std::shared_ptr<IVirtualCameraSource> virtualCameraSource,
                         std::unique_ptr<MediaSynchronizer> mediaSynchronizer,
                         int framePerSeconds,
                         float classifierRangeThreshold)
    : Thread()
    , audioSource_(std::move(audioSource))
    , audioSink_(std::move(audioSink))
    , positionSource_(std::move(positionSource))
    , videoInput_(std::move(videoInput))
    , videoOutput_(std::move(videoOutput))
    , virtualCameraSource_(virtualCameraSource)
    , mediaSynchronizer_(std::move(mediaSynchronizer))
    , framePerSeconds_(framePerSeconds)
    , classifierRangeThreshold_(classifierRangeThreshold)
{
    if (!audioSource_ || !audioSink_ || !positionSource_ || !videoInput_ || !videoOutput_ || !virtualCameraSource_)
    {
        throw std::invalid_argument("Error in MediaThread - Null is not a valid argument");
    }
}

/**
 * @brief Managing odas threads for audio and localization + camera and images processing.
 */
void MediaThread::run()
{
    FrameRateStabilizer frameStabilizer(framePerSeconds_);
    
    m_state = ThreadStatus::RUNNING;
    notify();

    // Start audio and video resources
    audioSource_->open();
    audioSink_->open();
    positionSource_->open();
    videoInput_->open();
    videoOutput_->open();

    std::cout << "MediaThread loop started" << std::endl;

    try
    {
        while (!isAbortRequested())
        {
            frameStabilizer.startFrame();

            Image image;
            while (videoInput_->readImage(image))
            {
                videoOutput_->writeImage(image);
            }

            AudioChunk audioChunk;
            while (audioSource_->readAudioChunk(audioChunk))
            {
                // Get audio sources and image spatial positions
                std::vector<SourcePosition> sourcePositions = positionSource_->getPositions();
                std::vector<VirtualCamera> virtualCameras = virtualCameraSource_->getVirtualCameras();

                if (virtualCameras.size() > 0)
                {
                    std::vector<SphericalAngleRect> imagePositions;
                    imagePositions.reserve(virtualCameras.size());
                    for (const auto& vc : virtualCameras)
                    {
                        imagePositions.push_back(vc);
                    }

                    std::vector<int> sourcesToKeep =
                                Classifier::getSourcesToKeep(sourcePositions, imagePositions, classifierRangeThreshold_);

                    AudioSuppresser::suppressNoise(sourcesToKeep, audioChunk);
                }

                audioSink_->write(audioChunk);
            }

            frameStabilizer.endFrame();
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "Error in media thread : " << e.what() << std::endl;
        m_state = ThreadStatus::CRASHED;
    }

    // Clean audio and video resources
    audioSource_->close();
    audioSink_->close();
    positionSource_->close();
    videoInput_->close();
    videoOutput_->close();

    std::cout << "MediaThread loop finished" << std::endl;
    
    if (m_state != ThreadStatus::CRASHED)
    {
        m_state = ThreadStatus::STOPPED;
    }
    
    notify();
}
}    // namespace Model

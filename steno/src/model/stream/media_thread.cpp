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
                         std::unique_ptr<MediaSynchronizer> mediaSynchronizer,
                         int framePerSeconds,
                         float classifierRangeThreshold)
    : Thread()
    , audioSource_(std::move(audioSource))
    , audioSink_(std::move(audioSink))
    , positionSource_(std::move(positionSource))
    , videoInput_(std::move(videoInput))
    , videoOutput_(std::move(videoOutput))
    , mediaSynchronizer_(std::move(mediaSynchronizer))
    , framePerSeconds_(framePerSeconds)
    , classifierRangeThreshold_(classifierRangeThreshold)
{
    if (!audioSource_ || !audioSink_ || !positionSource_ || !videoInput_ || !videoOutput_)
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

    // Start audio and video resources
    audioSource_->open();
    audioSink_->open();
    positionSource_->open();
    videoInput_->open();
    videoOutput_->open();

    std::cout << "MediaThread loop started" << std::endl;

    unsigned long long lastAudioTimeStamp = 0;
    unsigned long long lastImageTimeStamp = 0;

    try
    {
        while (!isAbortRequested())
        {
            frameStabilizer.startFrame();

            Image image;
            while (videoInput_->readImage(image))
            {
                videoOutput_->writeImage(image);

                std::cout << "image: " << image.timeStamp - lastImageTimeStamp << std::endl;
                lastImageTimeStamp = image.timeStamp;

                //mediaSynchronizer_->queueImage(image);
            }

            AudioChunk audioChunk;
            while (audioSource_->readAudioChunk(audioChunk))
            {
                audioSink_->write(audioChunk.audioData.get(), audioChunk.size);

                std::cout << "audio: " << audioChunk.timestamp - lastAudioTimeStamp << std::endl;
                lastAudioTimeStamp = audioChunk.timestamp;

                //mediaSynchronizer_->queueAudio(audioChunk);
            }

            /*SynchronizedMedia outputMedia;
            bool syncSuccess = mediaSynchronizer_->synchronize(outputMedia);
            if (syncSuccess)
            {
                AudioChunk& outputAudio = outputMedia.audioChunk;
                Image& outputImage = outputMedia.image;

                audioSink_->write(outputAudio.audioData.get(), outputAudio.size);

                if (outputMedia.hasImage)
                {
                    videoOutput_->writeImage(outputImage);
                }
            }*/

            frameStabilizer.endFrame();
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

    std::cout << "MediaThread loop stopped" << std::endl;
}
}    // namespace Model

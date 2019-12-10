#include "media_synchronizer.h"

#include <iostream>
#include <cmath>

namespace
{
const float ACCEPTABLE_DELAY_FRAMETIME_MULTIPLIER = 1.2f;
}

namespace Model
{

MediaSynchronizer::MediaSynchronizer(int frameTimeUs)
    : acceptableDelayUs_(frameTimeUs * ACCEPTABLE_DELAY_FRAMETIME_MULTIPLIER)
{
}

void MediaSynchronizer::queueAudio(AudioChunk audioChunk)
{
    audioQueue_.push(audioChunk);
}

void MediaSynchronizer::queueImage(Image image)
{
    imageQueue_.push(image);
}

bool MediaSynchronizer::synchronize(SynchronizedMedia& outMedia)
{
    outMedia.hasAudio = false;
    outMedia.hasImage = false;

    AudioChunk audio;
    if (!audioQueue_.empty())
    {
        audio = audioQueue_.front();
        outMedia.hasAudio = true;
    }

    Image image;
    if (!imageQueue_.empty())
    {
        image = imageQueue_.front();
        outMedia.hasImage = true;
    }

    if (outMedia.hasImage && outMedia.hasAudio)
    {
        long long timeDiff = static_cast<long long>(audio.timestamp - image.timeStamp);
        if (std::abs(timeDiff) > acceptableDelayUs_)
        {
            if (timeDiff < 0)
            {
                // video ahead of audio, we send the last frame
                outMedia.hasImage = false;
            }
            else
            {
                // audio ahead of video, we trash the video
                while(std::abs(timeDiff) > acceptableDelayUs_)
                {
                    imageQueue_.pop();

                    if (imageQueue_.empty())
                    {
                        outMedia.hasImage = false;
                        break;
                    }

                    image = imageQueue_.front();
                    timeDiff = static_cast<long long>(audio.timestamp - image.timeStamp);
                }
            }
        }
    }

    outMedia.audioChunk = audio;
    outMedia.image = image;

    if (outMedia.hasImage)
    {
        imageQueue_.pop();
    }

    if (outMedia.hasAudio)
    {
        audioQueue_.pop();
    }

    return true;
}

bool MediaSynchronizer::popAudio(AudioChunk& outAudioChunk)
{
    if (audioQueue_.empty())
    {
        return false;
    }

    outAudioChunk = audioQueue_.front();
    audioQueue_.pop();

    return true;
}

}    // namespace Model

#include "media_synchronizer.h"

#include <iostream>
#include <cmath>

namespace Model
{

MediaSynchronizer::MediaSynchronizer(int acceptableDelayMs)
    : acceptableDelayMs_(acceptableDelayMs)
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
    if (imageQueue_.empty() || audioQueue_.empty())
    {
        std::cout << "queue empty" << std::endl;
        return false;
    }

    std::cout << "imageQueue: " << imageQueue_.size() << ", audioQueue: " << audioQueue_.size() << std::endl;

    Image image = imageQueue_.front();
    outMedia.hasImage = true;

    AudioChunk audio = audioQueue_.front();

    long long timeDiff = static_cast<long long>(audio.timestamp - image.timeStamp);
    //std::cout << "timeDiff: " << timeDiff << std::endl;

    if (std::abs(timeDiff) > acceptableDelayMs_)
    {
        if (timeDiff < 0)
        {
            // video ahead of audio, we send the last frame
            std::cout << "MediaSynchronizer: video ahead of audio: " << timeDiff << std::endl;
            outMedia.hasImage = false;
        }
        else
        {
            // audio ahead of video, we trash the video
            std::cout << "===MediaSynchronizer: audio ahead of video: " << timeDiff << std::endl;
            while(std::abs(timeDiff) > acceptableDelayMs_)
            {
                imageQueue_.pop();

                if (imageQueue_.empty())
                {
                    std::cout << "image queue size: " << imageQueue_.size() << std::endl;
                    outMedia.hasImage = false;
                    break;
                }

                image = imageQueue_.front();
                timeDiff = static_cast<long long>(audio.timestamp - image.timeStamp);
                std::cout << "timeDiff: " << timeDiff << std::endl;
            }
        }
    }
    else
    {
        std::cout << "Synced!" << std::endl;
    }

    outMedia.audioChunk = audio;
    outMedia.image = image;

    if (!imageQueue_.empty() && outMedia.hasImage)
    {
        imageQueue_.pop();
    }

    if (!audioQueue_.empty())
    {
        audioQueue_.pop();
    }

    return true;
}

}    // namespace Model

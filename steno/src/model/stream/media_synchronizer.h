#ifndef MEDIA_SYNCHRONIZER_H
#define MEDIA_SYNCHRONIZER_H

#include <iostream>
#include <queue>
#include <vector>

#include "model/stream/synchronized_media.h"

namespace Model
{

class MediaSynchronizer
{
public:
    MediaSynchronizer(int acceptableDelayMs);
    ~MediaSynchronizer() = default;

    void queueAudio(AudioChunk audioChunk);
    void queueImage(Image image);
    bool synchronize(SynchronizedMedia& outMedia);
    
private:
    int acceptableDelayMs_;
    std::queue<AudioChunk> audioQueue_;
    std::queue<Image> imageQueue_;
};

}    // namespace Model

#endif    //! MEDIA_SYNCHRONIZER_H

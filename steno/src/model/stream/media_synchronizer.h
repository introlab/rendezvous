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
    MediaSynchronizer(int frameTimeUs);
    ~MediaSynchronizer() = default;

    void queueAudio(AudioChunk audioChunk);
    void queueImage(Image image);
    bool synchronize(SynchronizedMedia& outMedia);
    bool popAudio(AudioChunk& outAudioChunk);
    
private:
    int acceptableDelayUs_;
    std::queue<AudioChunk> audioQueue_;
    std::queue<Image> imageQueue_;
};

}    // namespace Model

#endif    //! MEDIA_SYNCHRONIZER_H

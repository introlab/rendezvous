#ifndef SYNCHRONIZED_MEDIA_H
#define SYNCHRONIZED_MEDIA_H

#include "model/stream/audio/audio_chunk.h"
#include "model/stream/utils/images/images.h"

namespace Model
{

struct SynchronizedMedia
{
    AudioChunk audioChunk;
    Image image;

    bool hasImage;
};

}    // namespace Model

#endif    //! SYNCHRONIZED_MEDIA_H
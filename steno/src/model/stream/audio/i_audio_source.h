#ifndef I_AUDIO_SOURCE_H
#define I_AUDIO_SOURCE_H

#include "model/stream/audio/audio_chunk.h"

namespace Model
{
class IAudioSource
{
   public:
    virtual ~IAudioSource() = default;

    virtual void open() = 0;
    virtual void close() = 0;
    virtual bool readAudioChunk(AudioChunk& outAudioChunk) = 0;
};

}    // namespace Model

#endif    // I_AUDIO_SOURCE_H

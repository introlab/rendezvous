#ifndef I_AUDIO_SINK_H
#define I_AUDIO_SINK_H

#include <cstdint>

#include "model/stream/audio/audio_chunk.h"

namespace Model
{
class IAudioSink
{
   public:
    virtual ~IAudioSink() = default;

    virtual void open() = 0;
    virtual void close() = 0;
    virtual int write(const AudioChunk& audioChunk) = 0;
};

}    // namespace Model

#endif    // I_AUDIO_SINK_H

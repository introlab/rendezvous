#ifndef AUDIO_SUPPRESSER_H
#define AUDIO_SUPPRESSER_H

#include <cstdint>
#include <vector>

#include "model/stream/audio/audio_chunk.h"

namespace Model
{
class AudioSuppresser
{
   public:
    static void suppressNoise(const std::vector<int> &indexToKeep, AudioChunk& audioChunk);
};

}    // namespace Model

#endif    // AUDIO_SUPPRESSER_H

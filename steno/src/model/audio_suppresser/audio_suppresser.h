#ifndef AUDIO_SUPPRESSER_H
#define AUDIO_SUPPRESSER_H

#include <cstdint>
#include <vector>

namespace Model
{
class AudioSuppresser
{
   public:
    static void suppressNoise(const std::vector<int> &indexToKeep, uint8_t *audioBuf, const int bufferLength);

   private:
    static void createMaskFromIndex(const int index, uint8_t *mask, const int maskLength);
};

}    // namespace Model

#endif    // AUDIO_SUPPRESSER_H

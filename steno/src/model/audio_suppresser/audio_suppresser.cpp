#include "audio_suppresser.h"

#include <cstring>

namespace Model
{
/**
 * @brief Removes all audio sources that are not in the vector sourcesToKeep
 * @param [IN] sourcesToKeep - vector of audio sources to keep.
 * @param [IN/OUT] audioBuf - Audio buffer to modify
 * @param [IN] bufferLength
 */
void AudioSuppresser::suppressNoise(const std::vector<int> &sourcesToKeep, AudioChunk& audioChunk)
{
    // Initialize mask to zeros
    uint8_t mask[audioChunk.channels] = {};

    // Build mask
    for (int channel : sourcesToKeep)
    {
        if (channel >= audioChunk.channels)
        {
            throw std::runtime_error("AudioSuppresser error : channel index is invalid!");
        }
        
        mask[channel] = 0xFF;
    }

    uint8_t* data = audioChunk.audioData.get();

    // Supress sources that we don't want to keep
    for (int i = 0; i < static_cast<int>(audioChunk.size); i++)
    {
        int channel = (i / audioChunk.bytesPerChannel) % audioChunk.channels;
        data[i] &= mask[channel];
    }
}

}    // namespace Model

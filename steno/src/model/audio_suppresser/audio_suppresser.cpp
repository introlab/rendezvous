#include "audio_suppresser.h"

namespace Model
{
/**
 * @brief Removes all audio sources that are not in the vector sourcesToKeep/
 * @param [IN] sourcesToKeep - vector of audio sources to keep.
 * @param [IN/OUT] audioBuf - Audio buffer to modify
 * @param [IN] bufferLength
 */
void AudioSuppresser::suppressNoise(const std::vector<int> &sourcesToKeep, uint8_t *audioBuf, const int bufferLength)
{
    // Initialize mask
    uint8_t mask[bufferLength];

    for (int i = 0; i < bufferLength; i++)
    {
        mask[i] = 0;
    }

    // Build mask
    int index;
    for (std::size_t i = 0; i < sourcesToKeep.size(); i++)
    {
        index = sourcesToKeep[i];
        createMaskFromIndex(index, mask, bufferLength);
    }

    // Supress sources that we don't want to keep
    for (int i = 0; i < bufferLength; i++)
    {
        audioBuf[i] = audioBuf[i] & mask[i];
    }
}

/**
 * @brief Create a mask to supress all audio sources that we don't want to keep.
 * @param [IN] index - index of the source to keep
 * @param [OUT] mask - mask for suppression
 * @param [IN] maskLength
 */
void AudioSuppresser::createMaskFromIndex(const int index, uint8_t *mask, const int maskLength)
{
    // By default we use 16 bits precision => 2 bytes per source
    for (int i = 2 * index; i < maskLength; i = i + 8)
    {
        mask[i] = 255;

        if (i + 1 < maskLength) mask[i + 1] = 255;
    }
}

}    // namespace Model

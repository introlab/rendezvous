#include "audio_suppresser.h"

namespace Model
{
/**
 * @brief Take audio buffer in input and remove sources in sourcesToSuppress/
 * @param [IN] sourcesToSuppress - vector of audio sources to remove.
 * @param [IN/OUT] audioBuf - Audio buffer to modify
 * @param [IN] bufferLength
 */
void AudioSuppresser::suppressSources(const std::vector<int> &sourcesToSuppress, uint8_t *audioBuf,
                                      const int bufferLength)
{
    // Initialize mask
    uint8_t mask[bufferLength];

    for (int i = 0; i < bufferLength; i++)
    {
        mask[i] = 255;
    }

    // Build mask
    int index;
    for (std::size_t i = 0; i < sourcesToSuppress.size(); i++)
    {
        index = sourcesToSuppress[i];
        createMaskFromIndex(index, mask, bufferLength);
    }

    // Supress sources
    for (int i = 0; i < bufferLength; i++)
    {
        audioBuf[i] = audioBuf[i] & mask[i];
    }
}

/**
 * @brief Create a mask to suppress a source in a audio source.
 * @param [IN] index - index of the source
 * @param [OUT] mask - mask for suppression
 * @param [IN] maskLength
 */
void AudioSuppresser::createMaskFromIndex(const int index, uint8_t *mask, const int maskLength)
{
    // By default we use 16 bits precision => 2 bytes per source
    for (int i = 2 * index; i < maskLength; i = i + 8)
    {
        mask[i] = 0;

        if (i + 1 < maskLength) mask[i + 1] = 0;
    }
}

}    // namespace Model

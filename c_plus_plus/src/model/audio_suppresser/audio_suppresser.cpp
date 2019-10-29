#include "audio_suppresser.h"

namespace Model
{

void AudioSuppresser::suppressSources(const std::vector<int> &sourcesToSuppress, uint8_t* audioBuf, const int bufferLength)
{
    // Initialize mask
    uint8_t mask[bufferLength];

    for(int i = 0; i < bufferLength; i++)
    {
        mask[i] = 255;
    }

    // Build mask
    int index;
    for(std::size_t i = 0; i < sourcesToSuppress.size(); i++)
    {
        index = sourcesToSuppress[i];
        createMaskFromIndex(index, mask, bufferLength);
    }

    // Supress sources
    for(int i = 0; i < bufferLength; i++)
    {
        audioBuf[i] = audioBuf[i] & mask[i];
    }
}

void AudioSuppresser::createMaskFromIndex(const int index, uint8_t *mask, const int maskLength)
{
    // By default we use 16 bits precision => 2 bytes per source
    for(int i = 2 * index; i < maskLength; i = i + 8)
    {
        mask[i] = 0;

        if(i + 1 < maskLength)
            mask[i+1] = 0;
    }
}

} // Model

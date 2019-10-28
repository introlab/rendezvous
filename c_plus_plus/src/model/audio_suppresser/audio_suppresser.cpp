#include "audio_suppresser.h"

namespace Model
{

void AudioSuppresser::suppressSources(std::vector<int> sourcesToSuppress, uint8_t* audioBuf, const int bufferLength)
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
        switch (index)
        {
        case 0: createMaskFromIndex(static_cast<int>(index), mask, bufferLength);
                break;
        case 1: createMaskFromIndex(static_cast<int>(index), mask, bufferLength);
                break;
        case 2: createMaskFromIndex(static_cast<int>(index), mask, bufferLength);
                break;
        case 3: createMaskFromIndex(static_cast<int>(index), mask, bufferLength);
                break;
        }
    }

    // Supress sources
    for(int i = 0; i < bufferLength; i++)
    {
        audioBuf[i] = audioBuf[i] & mask[i];
    }
}

void AudioSuppresser::createMaskFromIndex(int index, uint8_t *mask, int maskLength)
{
    // by default we use 16 bits precision => 2 bytes per sources
    for(int i = 2*index; i < maskLength; i = i + 8)
    {
        // TODO: insert white noise instead of 0
        mask[i] = 0;
        mask[i+1] = 0;
    }
}

} // Model

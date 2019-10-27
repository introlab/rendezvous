#include "audio_suppresser.h"

namespace Model
{

void AudioSuppresser::suppressSources(std::vector<int> sourcesToSuppress, uint8_t* audioBuf, uint8_t* audioBufSuppressed)
{
    const int bufferLenght = static_cast<int>(sizeof(audioBuf));

    // Initialize mask
    uint8_t mask[bufferLenght];

    for(int i = 0; i < bufferLenght; i++)
    {
        mask[i] = 255;
    }

    // Build mask
    for(size_t i = 0; i < sourcesToSuppress.size(); i++)
    {
        switch (sourcesToSuppress[i])
        {
        case 0: createMaskFromIndex(static_cast<int>(i), mask);
                break;
        case 1: createMaskFromIndex(static_cast<int>(i), mask);
                break;
        case 2: createMaskFromIndex(static_cast<int>(i), mask);
                break;
        case 3: createMaskFromIndex(static_cast<int>(i), mask);
                break;
        }
    }
}

void AudioSuppresser::createMaskFromIndex(int index, uint8_t *mask)
{
    int maskLength = static_cast<int>(sizeof(mask));

    // by default we use 16 bits precision => 2 bytes per sources
    for(int i = 2*index; i < maskLength; i = i + 8)
    {
        // TODO: insert white noise instead of 0
        mask[i] = 0;
        mask[i+1] = 0;
    }
}

} // Model

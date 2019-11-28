#ifndef AUDIO_CHUNK_H
#define AUDIO_CHUNK_H

#include <cstdint>
#include <memory>

namespace Model
{
struct AudioChunk
{
    AudioChunk() = default;
    AudioChunk(int elements, int channels, int bytesPerChannel)
        : size(elements * channels * bytesPerChannel)
        , channels(channels)
        , bytesPerChannel(bytesPerChannel)
        , timestamp(0)
        , audioData(nullptr)
    {
    }
    ~AudioChunk() = default;

    std::size_t size;
    int channels;
    int bytesPerChannel;
    unsigned long long timestamp;

    std::shared_ptr<uint8_t> audioData;
};

}    // namespace Model

#endif    //! AUDIO_CHUNK_H
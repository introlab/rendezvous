#ifndef I_AUDIO_SOURCE_H
#define I_AUDIO_SOURCE_H

#include <cinttypes>

namespace Model
{
class IAudioSource
{
   public:
    virtual ~IAudioSource() = default;

    virtual void open() = 0;
    virtual void close() = 0;
    virtual int read(uint8_t* audioBuf, int bytesToRead) = 0;
};

}    // namespace Model

#endif    // I_AUDIO_SOURCE_H

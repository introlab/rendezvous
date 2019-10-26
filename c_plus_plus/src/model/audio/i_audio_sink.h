#ifndef I_AUDIO_SINK_H
#define I_AUDIO_SINK_H

#include <cstdint>

namespace Model
{
class IAudioSink
{
   public:
    virtual ~IAudioSink() = default;
    virtual bool open() = 0;
    virtual bool close() = 0;
    virtual int write(uint8_t* buffer, int nbytes) = 0;
};

}    // Model

#endif    // I_AUDIO_SINK_H

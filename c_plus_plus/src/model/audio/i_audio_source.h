#ifndef I_AUDIO_SOURCE_H
#define I_AUDIO_SOURCE_H

#include <cinttypes>


namespace Model
{

class IAudioSource
{
public:
    virtual ~IAudioSource() = default;
    
    virtual bool open() = 0;
    virtual bool close() = 0;
    virtual int read(uint8_t* audioBuf, int nbytes)  = 0;
};

} // Model

#endif // I_AUDIO_SOURCE_H

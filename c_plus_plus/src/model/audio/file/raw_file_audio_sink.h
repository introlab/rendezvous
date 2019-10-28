#ifndef RAW_FILE_AUDIO_SINK_H
#define RAW_FILE_AUDIO_SINK_H

#include <string>

#include "model/audio/i_audio_sink.h"


namespace Model
{

class RawFileAudioSink : public IAudioSink
{
public:
    RawFileAudioSink(std::string fileName);
    ~RawFileAudioSink() override;

    bool open() override;
    bool close() override;
    int write(uint8_t* buffer, int bytesToWrite) override;

private:
    FILE* m_file;
    std::string m_fileName;
};

} // Model

#endif // RAW_FILE_AUDIO_SINK_H

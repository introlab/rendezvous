#ifndef RAW_FILE_AUDIO_SINK_H
#define RAW_FILE_AUDIO_SINK_H

#include <string>

#include "model/stream/audio/i_audio_sink.h"

namespace Model
{
class RawFileAudioSink : public IAudioSink
{
   public:
    RawFileAudioSink(const std::string& fileName);
    ~RawFileAudioSink() override;

    void open() override;
    void close() override;
    int write(uint8_t* buffer, int bytesToWrite) override;

   private:
    FILE* m_file;
    std::string m_fileName;
};

}    // namespace Model

#endif    // RAW_FILE_AUDIO_SINK_H

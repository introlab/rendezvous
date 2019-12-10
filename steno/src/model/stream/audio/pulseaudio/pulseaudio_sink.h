#ifndef I_PULSEAUDIO_SINK_H
#define I_PULSEAUDIO_SINK_H

#include <string>

#include <pulse/simple.h>

#include "model/stream/audio/audio_config.h"
#include "model/stream/audio/i_audio_sink.h"
#include "model/stream/utils/threads/thread.h"
#include "model/stream/utils/threads/readerwriterqueue.h"

namespace Model
{
class PulseAudioSink : public IAudioSink, protected Thread
{
   public:
    PulseAudioSink(std::shared_ptr<AudioConfig> audioConfig);
    ~PulseAudioSink() override;

    void open() override;
    void close() override;
    int write(const AudioChunk& audioChunk) override;

    void run() override;

   private:
    std::string m_deviceName;
    pa_simple* m_stream;
    pa_sample_spec m_ss{};

    moodycamel::ReaderWriterQueue<AudioChunk> outputAudioChunk_;
};

}    // namespace Model

#endif    // I_PULSEAUDIO_SINK_H

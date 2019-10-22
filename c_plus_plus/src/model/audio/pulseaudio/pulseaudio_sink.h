#ifndef I_PULSEAUDIO_SINK_H
#define I_PULSEAUDIO_SINK_H

#include <string>

#include <pulse/simple.h>

#include "model/audio/i_audio_sink.h"

namespace Model
{

class PulseAudioSink : public IAudioSink
{
public:
    PulseAudioSink(const std::string& device,
                   const uint8_t channels,
                   const uint32_t rate,
                   const pa_sample_format format);
    ~PulseAudioSink() override;

    bool open() override;
    bool close() override;
    int write(uint8_t* buffer, int nbytes) override;

private:
    std::string m_deviceName;
    pa_simple* m_stream;
    pa_sample_spec m_ss;
};

} // Model

#endif // I_PULSEAUDIO_SINK_H

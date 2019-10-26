#include "pulseaudio_sink.h"

#include <iostream>

#include <pulse/error.h>

namespace Model
{
PulseAudioSink::PulseAudioSink(std::string device, uint8_t channels,
                               uint32_t rate, pa_sample_format format)
    : m_deviceName(std::move(device)),
      m_stream(nullptr),
      m_ss({format, rate, channels})
{
}

PulseAudioSink::~PulseAudioSink() { close(); }
bool PulseAudioSink::open()
{
    if (m_stream != nullptr)
    {
        std::cout << "stream already initialized!" << std::endl;
        return false;
    }

    int error;
    m_stream = pa_simple_new(nullptr, "Fooapp", PA_STREAM_PLAYBACK,
                             m_deviceName.c_str(), "Music", &m_ss, nullptr,
                             nullptr, &error);

    if (m_stream == nullptr)
    {
        std::cout << "pa_simple_new() failed: " << pa_strerror(error)
                  << std::endl;
        return false;
    }

    return true;
}

bool PulseAudioSink::close()
{
    if (m_stream == nullptr) return true;

    int error;
    if (pa_simple_drain(m_stream, &error) < 0)
    {
        std::cout << "pa_simple_drain() failed: " << pa_strerror(error)
                  << std::endl;
        return false;
    }

    pa_simple_free(m_stream);
    m_stream = nullptr;

    return true;
}

int PulseAudioSink::write(uint8_t* buffer, int nbytes)
{
    int error;
    int bytesWritten =
        pa_simple_write(m_stream, buffer, static_cast<size_t>(nbytes), &error);
    if (bytesWritten < 0)
    {
        std::cout << "pa_simple_write() failed: " << pa_strerror(error)
                  << std::endl;
    }

    return bytesWritten;
}

}    // Model

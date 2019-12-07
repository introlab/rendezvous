#include "pulseaudio_sink.h"

#include <iostream>

#include <pulse/error.h>

namespace Model
{
PulseAudioSink::PulseAudioSink(std::shared_ptr<AudioConfig> audioConfig)
    : m_deviceName(audioConfig->deviceName)
    , m_stream(nullptr)
    , outputAudioChunk_(50)
{
    // default format is 16 bits little endian
    pa_sample_format sampleFormat = PA_SAMPLE_S16LE;
    switch (audioConfig->formatBytes)
    {
        case 8:
            sampleFormat = PA_SAMPLE_U8;
            break;
        case 16:
            sampleFormat = audioConfig->isLittleEndian ? PA_SAMPLE_S16LE : PA_SAMPLE_S16BE;
            break;
        case 32:
            sampleFormat = audioConfig->isLittleEndian ? PA_SAMPLE_S32LE : PA_SAMPLE_S32BE;
            break;
        default:
            break;
    }

    m_ss = {sampleFormat, static_cast<unsigned int>(audioConfig->rate), static_cast<uint8_t>(audioConfig->channels)};
}

PulseAudioSink::~PulseAudioSink()
{
    close();
}

void PulseAudioSink::open()
{
    if (m_stream != nullptr)
    {
        throw std::runtime_error("pulseaudio stream already initialized");
    }

    // dev of nullptr tells pulsaudio to use the default device
    const char* dev = nullptr;
    if (!m_deviceName.empty())
    {
        dev = m_deviceName.c_str();
    }

    int error;
    m_stream = pa_simple_new(nullptr, "record", PA_STREAM_PLAYBACK, dev, "record", &m_ss, nullptr, nullptr, &error);

    if (m_stream == nullptr)
    {
        throw std::runtime_error("cannot initialize pulseaudio stream: " + std::string(pa_strerror(error)));
    }

    start();
}

void PulseAudioSink::close()
{
    stop();
    join();

    if (m_stream == nullptr)
    {
        return;
    }

    pa_simple_flush(m_stream, nullptr);

    pa_simple_free(m_stream);
    m_stream = nullptr;
}

void PulseAudioSink::run()
{
    AudioChunk audioChunk;

    while (!isAbortRequested())
    {
        while (!outputAudioChunk_.try_dequeue(audioChunk) && !isAbortRequested())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        int error;
        int bytesWritten = pa_simple_write(m_stream, audioChunk.audioData.get(), audioChunk.size, &error);
        if (bytesWritten < 0)
        {
            throw std::runtime_error("pulseaudio write failed: " + std::string(pa_strerror(error)));
        }
    }
}

int PulseAudioSink::write(const AudioChunk& audioChunk)
{
    while (!outputAudioChunk_.try_enqueue(audioChunk) && !isAbortRequested()) {}

    std::cout << "Queue size " << outputAudioChunk_.size_approx() << std::endl;

    return audioChunk.size;
}

}    // namespace Model

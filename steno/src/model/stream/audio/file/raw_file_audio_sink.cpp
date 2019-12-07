#include "raw_file_audio_sink.h"

#include <iostream>

namespace Model
{
RawFileAudioSink::RawFileAudioSink(const std::string& fileName)
    : m_file(nullptr)
    , m_fileName(fileName)
{
}

RawFileAudioSink::~RawFileAudioSink()
{
    close();
}

void RawFileAudioSink::open()
{
    m_file = fopen(m_fileName.c_str(), "ab");
    if (m_file == nullptr)
    {
        throw std::runtime_error("cannot open file");
    }
}

void RawFileAudioSink::close()
{
    if (m_file != nullptr)
    {
        fclose(m_file);
        m_file = nullptr;
    }
}

int RawFileAudioSink::write(const AudioChunk& audioChunk)
{
    return fwrite(audioChunk.audioData.get(), sizeof(audioChunk.audioData.get()[0]), audioChunk.size, m_file);
}
}    // namespace Model

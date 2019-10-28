#include "raw_file_audio_sink.h"

namespace Model
{
RawFileAudioSink::RawFileAudioSink(std::string fileName)
    : m_file(nullptr)
    , m_fileName(std::move(fileName))
{
}

RawFileAudioSink::~RawFileAudioSink()
{
    close();
}

bool RawFileAudioSink::open()
{
    m_file = fopen(m_fileName.c_str(), "ab");
    return m_file != nullptr;
}

bool RawFileAudioSink::close()
{
    return fclose(m_file);
}

int RawFileAudioSink::write(uint8_t* buffer, int bytesToWrite)
{
    return fwrite(buffer, sizeof(buffer[0]), bytesToWrite, m_file);
}
}
#include "odas_audio_source.h"


namespace Model
{

OdasAudioSource::OdasAudioSource(quint16 port) :
    m_socketServer(std::make_unique<LocalSocketServer>(port))
{
}

OdasAudioSource::~OdasAudioSource()
{
    close();
}

bool OdasAudioSource::open()
{
    return m_socketServer->start();
}

bool OdasAudioSource::close()
{
    return m_socketServer->stop();
}

int OdasAudioSource::read(uint8_t* audioBuf, int bytesToRead)
{
    qint64 bytesRead = m_socketServer->read(reinterpret_cast<char*>(audioBuf), bytesToRead);
    return static_cast<int>(bytesRead);
}

} // Model

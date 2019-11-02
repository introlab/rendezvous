#include "odas_audio_source.h"

namespace Model
{
OdasAudioSource::OdasAudioSource(quint16 port)
    : m_socketServer(std::make_unique<LocalSocketServer>(port))
{
}

OdasAudioSource::~OdasAudioSource()
{
    close();
}

void OdasAudioSource::open()
{
    if (!m_socketServer->start())
    {
        throw std::runtime_error("cannot start socket server");
    }
}

void OdasAudioSource::close()
{
    m_socketServer->stop();
}

// TODO: broken because it is not possible to read from qt socket from
// another thread, will be fixed
int OdasAudioSource::read(uint8_t* /*audioBuf*/, int /*bytesToRead*/)
{
    return 0;
}

}    // namespace Model

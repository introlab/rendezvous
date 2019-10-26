#ifndef ODAS_AUDIO_SOURCE_H
#define ODAS_AUDIO_SOURCE_H

#include <memory>

#include "model/audio/i_audio_source.h"
#include "model/network/local_socket_server.h"

namespace Model
{
class OdasAudioSource : public IAudioSource
{
   public:
    OdasAudioSource(quint16 port);
    ~OdasAudioSource() override;

    bool open() override;
    bool close() override;
    int read(uint8_t* audioBuf, int bytesToRead) override;

   private:
    std::unique_ptr<LocalSocketServer> m_socketServer;
};

}    // namespace Model

#endif    // ODAS_AUDIO_SOURCE_H

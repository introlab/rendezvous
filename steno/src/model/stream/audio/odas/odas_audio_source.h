#ifndef ODAS_AUDIO_SOURCE_H
#define ODAS_AUDIO_SOURCE_H

#include <memory>

#include <QThread>

#include "model/stream/audio/audio_config.h"
#include "model/stream/audio/i_audio_source.h"
#include "model/stream/utils/models/circular_buffer.h"
#include "model/stream/utils/threads/readerwriterqueue.h"

namespace Model
{
class OdasAudioSource : public QThread, public IAudioSource
{
   public:
    OdasAudioSource(int port, int desiredChunkDurationMs, int numberOfBuffers, const AudioConfig& audioConfig);
    ~OdasAudioSource() override;

    void open() override;
    void close() override;
    bool readAudioChunk(AudioChunk& outAudioChunk) override;

   private:
    void run() override;
    unsigned long long calculateNewTimestamp(unsigned long long currentTimestamp, int bytesForward);

    int port_;
    AudioConfig audioConfig_;
    CircularBuffer<AudioChunk> audioChunks_;
    std::shared_ptr<moodycamel::BlockingReaderWriterQueue<AudioChunk>> audioQueue_;
};

}    // namespace Model

#endif    // ODAS_AUDIO_SOURCE_H

#include <cmath>

#include <QTcpServer>
#include <QTcpSocket>

#include "odas_audio_source.h"

namespace Model
{
OdasAudioSource::OdasAudioSource(int port, int desiredChunkDurationMs, int numberOfBuffers,
                                 const AudioConfig& audioConfig)
    : port_(port)
    , audioConfig_(audioConfig)
    , audioChunks_(numberOfBuffers, AudioChunk(desiredChunkDurationMs / 1000.f * audioConfig_.rate *
                                               audioConfig_.channels * audioConfig_.formatBytes))
    , audioQueue_(std::make_shared<moodycamel::BlockingReaderWriterQueue<AudioChunk>>(1))
{
    for (int i = 0; i < numberOfBuffers; i++)
    {
        audioChunks_.current().audioData =
            std::shared_ptr<uint8_t>(new uint8_t[audioChunks_.current().size], std::default_delete<uint8_t[]>());
        audioChunks_.next();
    }
}

OdasAudioSource::~OdasAudioSource()
{
    close();
}

void OdasAudioSource::open()
{
    start();
}

void OdasAudioSource::close()
{
    requestInterruption();
}

void OdasAudioSource::run()
{
    QTcpSocket* socket = nullptr;

    std::unique_ptr<QTcpServer> server = std::make_unique<QTcpServer>();
    server->setMaxPendingConnections(1);
    server->listen(QHostAddress::Any, port_);

    unsigned int readIndex = 0;
    while (!isInterruptionRequested())
    {
        if (socket == nullptr)
        {
            if (server->waitForNewConnection(1))
            {
                socket = server->nextPendingConnection();
            }
        }
        else
        {
            if (socket->state() == QAbstractSocket::ConnectedState)
            {
                if (socket->bytesAvailable() >= audioConfig_.packetAudioSize)
                {
                    AudioChunk& audioChunk = audioChunks_.current();

                    unsigned long long currentTimeStamp = 0;
                    socket->read(reinterpret_cast<char*>(&currentTimeStamp), audioConfig_.packetHeaderSize);

                    if (readIndex == 0)
                    {
                        audioChunk.timestamp = currentTimeStamp;
                    }

                    int remainingChunkSpace = audioChunk.size - readIndex;
                    int bytesToRead = remainingChunkSpace < audioConfig_.packetAudioSize ? remainingChunkSpace
                                                                                         : audioConfig_.packetAudioSize;

                    int firstBytesRead =
                        socket->read(reinterpret_cast<char*>(audioChunk.audioData.get()) + readIndex, bytesToRead);
                    readIndex += firstBytesRead;

                    // There is still data to be read if the chunk didn't have enough space
                    int remainingBytesToRead = audioConfig_.packetAudioSize - firstBytesRead;

                    // Current chunk is full
                    if (readIndex == audioChunk.size)
                    {
                        audioChunks_.next();
                        readIndex = 0;

                        // Start a new chunk if there is still data in the packet
                        if (remainingBytesToRead > 0)
                        {
                            AudioChunk& newAudioChunk = audioChunks_.current();

                            unsigned long long newTimeStamp = calculateNewTimestamp(currentTimeStamp, bytesToRead);
                            newAudioChunk.timestamp = newTimeStamp;

                            // Read remaining packet data in the new chunk
                            int secondBytesRead =
                                socket->read(reinterpret_cast<char*>(newAudioChunk.audioData.get()) + readIndex,
                                             remainingBytesToRead);
                            readIndex += secondBytesRead;
                        }

                        // Output audio chunk, if queue is full keep trying...
                        bool success = false;
                        while (!success && !isInterruptionRequested())
                        {
                            success = audioQueue_->try_enqueue(audioChunk);

                            if (!success)
                            {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }
                        }
                    }
                }
                else
                {
                    socket->waitForReadyRead(1);
                }
            }
            else
            {
                // reset read
                readIndex = 0;
                delete socket;
                socket = nullptr;
            }
        }
    }
}

bool OdasAudioSource::readAudioChunk(AudioChunk& outAudioChunk)
{
    return audioQueue_->wait_dequeue_timed(outAudioChunk, 500000);
}

unsigned long long OdasAudioSource::calculateNewTimestamp(unsigned long long currentTimestamp, int bytesForward)
{
    int numberOfSamples = bytesForward / audioConfig_.formatBytes / audioConfig_.channels;
    unsigned long long microseconds = numberOfSamples * std::pow(10, 6) / audioConfig_.rate;
    return currentTimestamp + microseconds;
}

}    // namespace Model

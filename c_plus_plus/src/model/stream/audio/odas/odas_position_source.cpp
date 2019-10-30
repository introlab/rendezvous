#include <QJsonArray>
#include <QJsonDocument>
#include <QMutexLocker>
#include <iostream>

#include "odas_position_source.h"

namespace Model
{
OdasPositionSource::OdasPositionSource(quint16 port)
    : m_socketServer(std::make_unique<LocalSocketServer>(port))
    , m_mutex(std::make_unique<QMutex>())
{
    connect(m_socketServer.get(), SIGNAL(dataReady(int)), this, SLOT(onPositionsReady(int)));
}

OdasPositionSource::~OdasPositionSource()
{
    close();
}

void OdasPositionSource::open()
{
    if (!m_socketServer->start())
    {
        throw std::runtime_error("cannot start socket server");
    }
}

void OdasPositionSource::close()
{
    m_socketServer->stop();
}

std::vector<SourcePosition> OdasPositionSource::getPositions()
{
    QMutexLocker locker(m_mutex.get());
    return m_sourcePositions;
}

void OdasPositionSource::updatePositions(std::vector<SourcePosition>& sourcePositions)
{
    QMutexLocker locker(m_mutex.get());
    m_sourcePositions = sourcePositions;
}

void OdasPositionSource::onPositionsReady(int numberOfBytes)
{
    std::vector<SourcePosition> sourcePositions;

    int bufferMaxSize = m_buffer.max_size();
    int maxBytesToRead = bufferMaxSize < numberOfBytes ? bufferMaxSize : numberOfBytes;
    int bytesRead = m_socketServer->read(m_buffer.data(), maxBytesToRead);
    QByteArray byteArray = QByteArray::fromRawData(m_buffer.data(), bytesRead);
    QJsonDocument json = QJsonDocument::fromJson(byteArray);

    QJsonArray odasSources = json["src"].toArray();
    for (auto it = odasSources.begin(); it < odasSources.end(); it++)
    {
        SourcePosition source = SourcePosition::deserialize(*it);
        sourcePositions.push_back(source);
    }

    updatePositions(sourcePositions);
}

}    // namespace Model

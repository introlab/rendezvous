#include <QJsonArray>
#include <QJsonDocument>
#include <QMutexLocker>

#include "odas_position_source.h"
#include "model/utils/spherical_angle_converter.h"


namespace Model
{

OdasPositionSource::OdasPositionSource(quint16 port) :
    m_socketServer(std::make_unique<LocalSocketServer>(port))
{
    connect(m_socketServer.get(), SIGNAL(dataReady(int)), this, SLOT(onPositionsReady(int)));
}

OdasPositionSource::~OdasPositionSource()
{
    close();
}

bool OdasPositionSource::open()
{
    return m_socketServer->start();
}

bool OdasPositionSource::close()
{
    return m_socketServer->stop();
}

std::vector<SourcePosition> OdasPositionSource::getPositions()
{
    QMutexLocker locker(&m_mutex);
    return m_sourcePositions;
}

void OdasPositionSource::updatePositions(std::vector<SourcePosition> sources)
{
    QMutexLocker locker(&m_mutex);
    m_sourcePositions = sources;
}

void OdasPositionSource::onPositionsReady(int numberOfBytes)
{
    std::vector<SourcePosition> sources;

    m_socketServer->read(m_buffer, numberOfBytes);
    QByteArray byteArray = QByteArray::fromRawData(m_buffer, numberOfBytes);
    QJsonDocument json = QJsonDocument::fromJson(byteArray);

    QJsonArray odasSources = json["src"].toArray();
    for (auto it = odasSources.begin(); it < odasSources.end(); it++)
    {
        SourcePosition source = SourcePosition::deserialize(*it);
        sources.push_back(source);
    }

    updatePositions(sources);
}

} // Model

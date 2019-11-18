#include "odas_position_source.h"

#include <iostream>

#include <QJsonArray>
#include <QJsonDocument>
#include <QMutexLocker>
#include <QTcpServer>
#include <QTcpSocket>

#include "model/stream/audio/source_position.h"

namespace Model
{
OdasPositionSource::OdasPositionSource(int port, int positionBufferSize)
    : m_port(port)
    , m_mutex(std::make_unique<QMutex>())
    , m_readBuffer(std::shared_ptr<char>(new char[positionBufferSize], std::default_delete<char[]>()))
{
}

OdasPositionSource::~OdasPositionSource()
{
    close();
}

void OdasPositionSource::open()
{
    start();
}

void OdasPositionSource::close()
{
    requestInterruption();
}

void OdasPositionSource::run()
{
    QTcpSocket* socket = nullptr;

    std::unique_ptr<QTcpServer> server = std::make_unique<QTcpServer>();
    server->setMaxPendingConnections(1);
    server->listen(QHostAddress::Any, m_port);

    std::cout << "Odas position source thread started" << std::endl;

    char buffer[100000];

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
                if (socket->bytesAvailable() > 0)
                {
                    int bytes = socket->bytesAvailable();
                    int bytesRead = socket->read(buffer, bytes);

                    std::vector<SourcePosition> sourcePositions;

                    QByteArray byteArray = QByteArray::fromRawData(buffer, bytesRead);
                    QJsonDocument json = QJsonDocument::fromJson(byteArray);

                    QJsonArray odasSources = json["src"].toArray();
                    for (auto it = odasSources.begin(); it < odasSources.end(); it++)
                    {
                        SourcePosition source = SourcePosition::deserialize(*it);
                        sourcePositions.push_back(source);
                    }

                    updatePositions(sourcePositions);
                }
                else
                {
                    socket->waitForReadyRead(1);
                }
            }
            else
            {
                // reset read
                delete socket;
                socket = nullptr;
            }
        }
    }

    std::cout << "Odas position source thread stopped" << std::endl;
}

std::vector<SourcePosition> OdasPositionSource::getPositions()
{
    QMutexLocker locker(m_mutex.get());
    return m_sourcePositions;
}

void OdasPositionSource::updatePositions(std::vector<SourcePosition>& positions)
{
    QMutexLocker locker(m_mutex.get());
    m_sourcePositions = positions;
}

}    // namespace Model

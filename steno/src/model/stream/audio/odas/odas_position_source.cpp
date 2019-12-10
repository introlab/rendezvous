#include "odas_position_source.h"

#include <QDebug>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMutexLocker>
#include <QTcpServer>
#include <QTcpSocket>

#include "model/stream/audio/source_position.h"

namespace
{
const int POSITION_SOURCE_BUFFER_SIZE = 10000;
}

namespace Model
{
OdasPositionSource::OdasPositionSource(int port)
    : m_port(port)
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

    qDebug() << "Odas position source thread started";

    char buffer[POSITION_SOURCE_BUFFER_SIZE];

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
                    if (!json.isNull())
                    {
                        QJsonValue odasSources = json["src"];
                        if (odasSources != QJsonValue::Undefined)
                        {
                            QJsonArray odasSourcesArray = odasSources.toArray();
                            for (auto it = odasSourcesArray.begin(); it < odasSourcesArray.end(); it++)
                            {
                                SourcePosition source = SourcePosition::deserialize(*it);
                                sourcePositions.push_back(source);
                            }
                        }
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

    qDebug() << "Odas position source thread stopped";
}

std::vector<SourcePosition> OdasPositionSource::getPositions()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_sourcePositions;
}

void OdasPositionSource::updatePositions(std::vector<SourcePosition>& positions)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_sourcePositions = positions;
}

}    // namespace Model

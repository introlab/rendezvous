#include "local_socket_server.h"

namespace Model
{
LocalSocketServer::LocalSocketServer(int port)
    : m_server(nullptr)
    , m_socket(nullptr)
    , m_port(port)
{
}

LocalSocketServer::~LocalSocketServer()
{
    stop();
}

bool LocalSocketServer::start()
{
    m_server = new QTcpServer();
    connect(m_server, SIGNAL(newConnection()), this, SLOT(onNewConnection()));

    if (m_server->isListening()) return true;

    return m_server->listen(QHostAddress::Any, m_port);
}

bool LocalSocketServer::stop()
{
    if (m_socket != nullptr)
    {
        m_socket->close();
        delete m_socket;
        m_socket = nullptr;
    }

    if (m_server != nullptr)
    {
        m_server->close();
        return m_server->isListening();
    }

    return true;
}

int LocalSocketServer::read(char* buffer, int bytesToRead)
{
    if (m_socket == nullptr || m_socket->state() != QAbstractSocket::ConnectedState)
    {
        return -1;
    }

    return m_socket->read(buffer, bytesToRead);
}

void LocalSocketServer::onNewConnection()
{
    if (m_socket != nullptr && m_socket->state() == QAbstractSocket::ConnectedState)
    {
        return;
    }

    m_socket = m_server->nextPendingConnection();

    connect(m_socket, &QTcpSocket::readyRead, this, [&] { emit dataReady(m_socket->bytesAvailable()); });
}

}    // namespace Model

#ifndef LOCAL_SOCKET_SERVER_H
#define LOCAL_SOCKET_SERVER_H

#include <QPointer>
#include <QTcpSocket>
#include <QTcpServer>

#include "i_socket_server.h"


namespace Model
{

class LocalSocketServer : public ISocketServer
{
Q_OBJECT

public:
    LocalSocketServer(int port);
    ~LocalSocketServer() override;

    bool start() override;
    bool stop() override;
    int read(char* buffer, int bytesToRead) override;

private slots:
    void onNewConnection();
    void onSocketStateChanged(QAbstractSocket::SocketState);
    
private:
    QPointer<QTcpServer> m_server;
    QPointer<QTcpSocket> m_socket;
    int m_port;
};

} // Model

#endif // LOCAL_SOCKET_SERVER_H
#ifndef I_SOCKET_SERVER_H
#define I_SOCKET_SERVER_H

#include <QObject>


namespace Model
{

class ISocketServer : public QObject
{
Q_OBJECT

public:
    virtual ~ISocketServer() = default;
    
    virtual bool start() = 0;
    virtual bool stop() = 0;
    virtual int read(char* buffer, int bytesToRead) = 0;

signals:
    void dataReady(int bytes);
};

} // Model

#endif // I_SOCKET_SERVER_H
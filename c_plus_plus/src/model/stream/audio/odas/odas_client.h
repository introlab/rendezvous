#ifndef ODAS_CLIENT_H
#define ODAS_CLIENT_H

#include <QThread>

namespace Model
{
class OdasClient : public QThread
{
    Q_OBJECT

   public:
    void stop();

   private:
    void run() override;
    bool m_isRunning = false;
    const int m_waitTime = 100;
    const int m_joinTime = 500;
};
}

#endif    // ODAS_CLIENT_H

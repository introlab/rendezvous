#ifndef ODAS_CLIENT_H
#define ODAS_CLIENT_H

#include <QThread>

namespace Model
{
class OdasClient : public QThread
{
    Q_OBJECT

   public:
    void stop() { m_isRunning = false; }
   private:
    void run() override;
    bool m_isRunning = false;
};
}

#endif    // ODAS_CLIENT_H

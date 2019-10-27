#ifndef ODAS_CLIENT_H
#define ODAS_CLIENT_H

#include <QProcess>
#include <QThread>

namespace Model
{
class OdasClient : public QThread
{
    Q_OBJECT

   public:
    OdasClient();
    void stop() { m_isRunning = false; }
   private:
    void run() override;

    std::unique_ptr<QProcess> m_pProcess;
    bool m_isRunning = false;
};
}

#endif    // ODAS_CLIENT_H

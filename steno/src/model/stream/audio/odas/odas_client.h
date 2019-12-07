#ifndef ODAS_CLIENT_H
#define ODAS_CLIENT_H

#include "model/app_config.h"
#include "model/stream/utils/threads/thread.h"
#include "model/utils/observer/subject.h"

#include <QProcess>
#include <vector>

namespace Model
{
class OdasClient : public Thread, public Subject
{
   public:
    OdasClient(std::shared_ptr<AppConfig> appConfig);

   protected:
    void run() override;

   private:
    void closeProcess(QProcess &process);

    const int m_waitTime = 100;    // ms
    const int m_joinTime = 500;    // ms
    std::shared_ptr<AppConfig> m_appConfig;
};
}    // namespace Model

#endif    // ODAS_CLIENT_H

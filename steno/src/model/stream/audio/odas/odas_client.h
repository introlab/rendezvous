#ifndef ODAS_CLIENT_H
#define ODAS_CLIENT_H

#include "model/app_config.h"
#include "model/stream/utils/threads/thread.h"
#include "model/utils/observer/i_subject.h"

#include <QProcess>
#include <vector>

namespace Model
{
enum class OdasClientState
{
    RUNNING,
    STOPPED,
    CRASHED
};

class OdasClient : public Thread, public ISubject
{
   public:
    OdasClient(std::shared_ptr<AppConfig> appConfig);
    void notify() override;
    void attach(IObserver *observer) override;
    void detach(IObserver *observer) override;
    OdasClientState getState();

   protected:
    void run() override;

   private:
    void closeProcess(QProcess &process);

    const int m_waitTime = 100;    // ms
    const int m_joinTime = 500;    // ms
    std::shared_ptr<AppConfig> m_appConfig;
    OdasClientState m_state = OdasClientState::STOPPED;
    std::vector<IObserver *> m_subscribers;
};
}    // namespace Model

#endif    // ODAS_CLIENT_H

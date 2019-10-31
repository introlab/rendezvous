#ifndef ODAS_CLIENT_H
#define ODAS_CLIENT_H

#include "model/stream/utils/threads/thread.h"
#include "model/utils/observer/i_subject.h"

#include <vector>

namespace Model
{
enum OdasClientState
{
    RUNNING,
    STOPPED
};

class OdasClient : public Thread, public ISubject
{
   public:
    void stop();
    void notify() override;
    void attach(IObserver *observer) override;
    OdasClientState getState();

   protected:
    void run() override;

   private:
    const int m_waitTime = 100;
    const int m_joinTime = 500;
    OdasClientState m_state = OdasClientState::STOPPED;
    std::vector<IObserver *> m_subscribers;
};
}

#endif    // ODAS_CLIENT_H

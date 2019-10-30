#ifndef ODAS_CLIENT_H
#define ODAS_CLIENT_H

#include "model/utils/observer/i_subject.h"

#include <QThread>

#include <vector>

namespace Model
{
enum OdasClientState
{
    RUNNING,
    STOPPED
};

class OdasClient : public QThread, public ISubject
{
    Q_OBJECT

   public:
    void stop();
    void notify() override;
    void attach(IObserver *observer) override;
    OdasClientState getState();

   signals:
    void stateChanged(const OdasClientState state);

   private:
    void run() override;
    const int m_waitTime = 100;
    const int m_joinTime = 500;
    OdasClientState m_state = OdasClientState::STOPPED;
    std::vector<IObserver *> m_subscribers;
};
}

#endif    // ODAS_CLIENT_H

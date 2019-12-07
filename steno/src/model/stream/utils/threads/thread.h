#ifndef THREAD_H
#define THREAD_H

#include <atomic>
#include <memory>
#include <thread>

namespace Model
{
class Thread
{
   public:
    Thread();
    virtual ~Thread();

    void start();
    void stop();
    void join();
    bool isRunning();
    bool isAbortRequested();
    void sleep(const int timeMs);

    enum class ThreadStatus
    {
        RUNNING,
        STOPPED,
        CRASHED
    };

    ThreadStatus getState()
    {
        return m_state;
    }

   protected:
    virtual void run() = 0;
    ThreadStatus m_state = ThreadStatus::STOPPED;

   private:
    void threadExecution();

    std::unique_ptr<std::thread> thread_;
    std::atomic<bool> isAbortRequested_;
    std::atomic<bool> isRunning_;
};

}    // namespace Model

#endif    //! THREAD_H

#ifndef THREAD_H
#define THREAD_H

#include <thread>
#include <atomic>
#include <memory>

namespace Model
{

class Thread
{
public:

    Thread();
    virtual ~Thread() = default;

    void start();
    void stop();
    void join();
    bool isRunning();
    bool isAbortRequested();

protected:

    virtual void run() = 0;

private:

    void threadExecution();

    std::unique_ptr<std::thread> thread_;
    std::atomic<bool> isAbortRequested_;
    std::atomic<bool> isRunning_;

};

} // Model

#endif //!THREAD_H

#include "thread.h"

#include <iostream>

namespace Model
{
Thread::Thread()
    : thread_(nullptr)
    , isAbortRequested_(false)
    , isRunning_(false)
{
}

Thread::~Thread()
{
    stop();
    join();
}

void Thread::start()
{
    if (!isRunning_)
    {
        join();    // Make sure the thread is truly finished

        thread_ = std::make_unique<std::thread>(&Thread::threadExecution, this);
        isRunning_ = true;
    }
}

void Thread::stop()
{
    if (isRunning_)
    {
        isAbortRequested_ = true;
    }
}

void Thread::join()
{
    if (thread_ != nullptr && thread_->joinable())
    {
        thread_->join();
    }
}

bool Thread::isRunning()
{
    return isRunning_;
}

bool Thread::isAbortRequested()
{
    return isAbortRequested_;
}

void Thread::threadExecution()
{
    try
    {
        run();
    }
    catch (const std::exception& e)
    {
        std::cout << "Exception during thread execution : " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "Unknown exception during thread execution" << std::endl;
    }

    isAbortRequested_ = false;
    isRunning_ = false;
}

/**
 * @brief Put the thread in pause for a certain time.
 * @param timeMs - time in milliseconds to sleep.
 */
void Thread::sleep(const int timeMs)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(timeMs));
}
}    // namespace Model

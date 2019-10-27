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

void Thread::start()
{
    if (!isRunning_)
    {
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
    if (isRunning_)
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
    catch (...)
    { 
        std::cout << "Unknown exception during thread execution" << std::endl;
    }

    isRunning_ = false;
}

} // Model

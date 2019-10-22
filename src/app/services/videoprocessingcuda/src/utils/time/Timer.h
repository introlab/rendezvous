#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer
{
public:

    void reset()
    {
        start_ = std::chrono::steady_clock::now();
    }
    
    template<typename Precision>
    uint64_t getElapsedTime()
    {
        auto time = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<Precision>(time - start_).count();
    }

private:

    std::chrono::time_point<std::chrono::_V2::steady_clock, std::chrono::nanoseconds> start_;

};

#endif //!TIMER_H
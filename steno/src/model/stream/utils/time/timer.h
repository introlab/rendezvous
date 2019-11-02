#ifndef TIMER_H
#define TIMER_H

#include <chrono>

namespace Model
{
class Timer
{
   public:
    void reset()
    {
        start_ = std::chrono::steady_clock::now();
    }

    template <typename Precision>
    uint64_t getElapsedTime()
    {
        auto time = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<Precision>(time - start_).count();
    }

   private:
    std::chrono::time_point<std::chrono::_V2::steady_clock, std::chrono::nanoseconds> start_;
};

}    // namespace Model

#endif    //! TIMER_H

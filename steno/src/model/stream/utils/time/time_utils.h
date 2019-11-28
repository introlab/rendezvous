#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include <chrono>

namespace Model
{
uint64_t systemTimeSinceEpoch()
{
    const auto now = std::chrono::system_clock::now();
    const auto epoch = now.time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(epoch).count());
}

uint64_t steadyTimeSinceEpoch()
{
    const auto now = std::chrono::steady_clock::now();
    const auto epoch = now.time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(epoch).count());
}

}    // namespace Model

#endif    //! TIME_UTILS_H
#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include <chrono>

namespace Model
{
uint64_t systemTimeSinceEpoch();

uint64_t steadyTimeSinceEpoch();

}    // namespace Model

#endif    //! TIME_UTILS_H
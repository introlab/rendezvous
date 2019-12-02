#ifndef HELPERS_H
#define HELPERS_H

#include <cmath>

#include "model/stream/utils/models/point.h"

namespace Model
{
namespace math
{
template <typename T>
T clamp(T value, const T& min, const T& max)
{
    if (value < min) value = min;
    if (value > max) value = max;
    return value;
}

template <typename T>
int sign(T val)
{
    return (T(0) < val) - (val < T(0));
}

template <typename T>
float euclideanDistance(T dx, T dy)
{
    return std::sqrt(dx * dx + dy * dy);
}

template <typename T>
float euclideanDistance(Point<T> d)
{
    return std::sqrt(d.x * d.x + d.y * d.y);
}

}    // namespace math

}    // namespace Model

#endif    //! HELPERS_H

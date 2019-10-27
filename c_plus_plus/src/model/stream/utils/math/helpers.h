#ifndef HELPERS_H
#define HELPERS_H

namespace Model
{

namespace math
{

template<typename T>
T clamp(T value, const T& min, const T& max)
{
    if (value < min) value = min;
    if (value > max) value = max;
    return value;
}

template <typename T> int sign(T val)
{
    return (T(0) < val) - (val < T(0));
}

}



} // Model

#endif //!HELPERS_H

#ifndef DIM3_H
#define DIM3_H

#include "Dim2.h"

template<typename T>
struct Dim3 : public Dim2<T>
{
    Dim3() = default;

    Dim3(T width, T height, T channels)
        : Dim2<T>(width, height)
        , channels(channels)
    {
    }

    Dim3(const Dim2<T>& dim, T channels)
        : Dim2<T>(dim)
        , channels(channels)
    {
    }

    T channels;
};

template<typename T>
bool operator== (const Dim3<T> &lhs, const Dim3<T> &rhs)
{
    return lhs.channels == rhs.channels && static_cast<Dim2<T>>(lhs) == static_cast<Dim2<T>>(rhs);
}

template<typename T>
bool operator!= (const Dim3<T> &lhs, const Dim3<T> &rhs)
{
    return !(lhs == rhs);
}

#endif //!DIM3_H

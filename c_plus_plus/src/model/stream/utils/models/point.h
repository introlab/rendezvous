#ifndef POINT_H
#define POINT_H

namespace Model
{

template<typename T>
struct Point
{
    Point() = default;
    Point(T x, T y)
        : x(x)
        , y(y)
    {
    }

    T x;
    T y;
};

} // Model

#endif //!POINT_H

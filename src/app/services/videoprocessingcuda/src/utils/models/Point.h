#ifndef POINT_H
#define POINT_H

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

#endif //!POINT_H
#ifndef DIM2_H
#define DIM2_H

namespace Model
{

template<typename T>
struct Dim2
{
    Dim2() = default;
    Dim2(T width, T height)
        : width(width)
        , height(height)
    {
    }

    T width;
    T height;
};

template<typename T>
bool operator== (const Dim2<T> &lhs, const Dim2<T> &rhs)
{
    return lhs.width == rhs.width && lhs.height == rhs.height;
}

template<typename T>
bool operator!= (const Dim2<T> &lhs, const Dim2<T> &rhs)
{
    return !(lhs == rhs);
}

} // Model

#endif //!DIM2_H

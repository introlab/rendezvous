#ifndef RECTANGLE_H
#define RECTANGLE_H

namespace Model
{

struct Rectangle
{
    Rectangle() = default;
    Rectangle(int x, int y, int width, int height)
        : x(x)
        , y(y)
        , width(width)
        , height(height)
    {
    }

    int x;
    int y;
    int width;
    int height;
};

} // Model

#endif //!RECTANGLE_H

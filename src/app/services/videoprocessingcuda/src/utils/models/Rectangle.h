#ifndef RECTANGLE_H
#define RECTANGLE_H

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

#endif //!RECTANGLE_H
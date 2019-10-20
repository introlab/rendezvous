#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

struct BoundingBox
{
    BoundingBox() = default;
    BoundingBox(int leftX, int rightX, int bottomY, int topY)
        : leftX(leftX)
        , rightX(rightX)
        , bottomY(bottomY)
        , topY(topY)
    {
    }

    int leftX;
    int rightX;
    int bottomY;
    int topY;
};

#endif //!BOUNDING_BOX_H
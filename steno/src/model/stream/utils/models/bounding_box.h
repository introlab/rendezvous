#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

namespace Model
{
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

}    // namespace Model

#endif    //! BOUNDING_BOX_H

#ifndef DONUT_SLICE_H
#define DONUT_SLICE_H

namespace Model
{
struct DonutSlice
{
    DonutSlice() = default;
    DonutSlice(float xCenter, float yCenter, float inRadius, float outRadius, float middleAngle, float angleSpan)
        : xCenter(xCenter)
        , yCenter(yCenter)
        , inRadius(inRadius)
        , outRadius(outRadius)
        , middleAngle(middleAngle)
        , angleSpan(angleSpan)
    {
    }

    float xCenter;
    float yCenter;
    float inRadius;
    float outRadius;
    float middleAngle;
    float angleSpan;
};

}    // namespace Model

#endif    // !DONUT_SLICE_H

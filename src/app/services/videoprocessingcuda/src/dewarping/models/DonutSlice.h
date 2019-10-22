#ifndef DONUT_SLICE_H
#define DONUT_SLICE_H

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

#endif // !DONUT_SLICE_H

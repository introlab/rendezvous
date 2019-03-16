#ifndef DONUT_SLICE_H
#define DONUT_SLICE_H

struct DonutSlice
{
public:

    DonutSlice(float xCenter, float yCenter, float inRadius, float outRadius, float middleAngle, float angleSpan);

    float xCenter;
    float yCenter;
    float inRadius;
    float outRadius;
    float middleAngle;
    float angleSpan;
};

#endif // !DONUT_SLICE_H


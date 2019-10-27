#ifndef DEWAPING_PARAMETERS_H
#define DEWAPING_PARAMETERS_H

namespace Model
{

struct DewarpingParameters
{

    DewarpingParameters() = default;
    DewarpingParameters(float xCenter, float yCenter, float dewarpWidth, float dewarpHeight, float inRadius, float centerRadius,
                        float outRadiusDiff, float xOffset, float bottomDistorsionFactor, float topOffset = 0, float bottomOffset = 0)
        : xCenter(xCenter)
        , yCenter(yCenter)
        , dewarpWidth(dewarpWidth)
        , dewarpHeight(dewarpHeight)
        , inRadius(inRadius)
        , centerRadius(centerRadius)
        , outRadiusDiff(outRadiusDiff)
        , xOffset(xOffset)
        , bottomDistorsionFactor(bottomDistorsionFactor)
        , topOffset(topOffset)
        , bottomOffset(bottomOffset)
    {
    }

    float xCenter;
    float yCenter;
    float dewarpWidth;
    float dewarpHeight;
    float inRadius;
    float centerRadius;
    float outRadiusDiff;
    float xOffset;
    float bottomDistorsionFactor;
    float topOffset;
    float bottomOffset;

};

} // Model

#endif // !DEWAPING_PARAMETERS_H


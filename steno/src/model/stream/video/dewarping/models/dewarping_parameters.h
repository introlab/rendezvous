#ifndef DEWAPING_PARAMETERS_H
#define DEWAPING_PARAMETERS_H

namespace Model
{
struct DewarpingParameters
{
    DewarpingParameters() = default;
    DewarpingParameters(float xCenter, float yCenter, float dewarpWidth, float dewarpHeight, float inRadius,
                        float centerRadius, float outRadiusDiff, float xOffset, float bottomDistorsionFactor)
        : xCenter(xCenter)
        , yCenter(yCenter)
        , dewarpWidth(dewarpWidth)
        , dewarpHeight(dewarpHeight)
        , inRadius(inRadius)
        , centerRadius(centerRadius)
        , outRadiusDiff(outRadiusDiff)
        , xOffset(xOffset)
        , bottomDistorsionFactor(bottomDistorsionFactor)
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
};

}    // namespace Model

#endif    // !DEWAPING_PARAMETERS_H

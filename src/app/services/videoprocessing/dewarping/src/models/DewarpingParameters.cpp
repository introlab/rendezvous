#include <models/DewarpingParameters.h>

DewarpingParameters::DewarpingParameters()
    : xCenter(0),
    yCenter(0),
    dewarpWidth(0),
    dewarpHeight(0),
    inRadius(0),
    centerRadius(0),
    outRadiusDiff(0),
    xOffset(0),
    bottomDistorsionFactor(0)
{
}

DewarpingParameters::DewarpingParameters(float xCenter, float yCenter, float dewarpWidth, float dewarpHeight,
    float inRadius, float centerRadius, float outRadiusDiff, float xOffset, float bottomDistorsionFactor)
    : xCenter(xCenter),
    yCenter(yCenter),
    dewarpWidth(dewarpWidth),
    dewarpHeight(dewarpHeight),
    inRadius(inRadius),
    centerRadius(centerRadius),
    outRadiusDiff(outRadiusDiff),
    xOffset(xOffset),
    bottomDistorsionFactor(bottomDistorsionFactor)
{
}

#include "DewarpingParameters.h"

DewarpingParameters::DewarpingParameters(float xCenter, float yCenter, float dewarpWidth, float dewarpHeight, float inRadius,
    float centerRadius, float outRadiusDiff, float xOffset, float bottomDistorsionFactor, float topOffset, float bottomOffset)
    : xCenter(xCenter),
    yCenter(yCenter),
    dewarpWidth(dewarpWidth),
    dewarpHeight(dewarpHeight),
    inRadius(inRadius),
    centerRadius(centerRadius),
    outRadiusDiff(outRadiusDiff),
    xOffset(xOffset),
    bottomDistorsionFactor(bottomDistorsionFactor),
    topOffset(topOffset),
    bottomOffset(bottomOffset)
{
}
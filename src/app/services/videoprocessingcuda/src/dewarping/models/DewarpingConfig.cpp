#include "DewarpingConfig.h"

DewarpingConfig::DewarpingConfig(float inRadius, float outRadius, float angleSpan,
                                 float topDistorsionFactor, float bottomDistorsionFactor, float fisheyeAngle)
    : inRadius(inRadius)
    , outRadius(outRadius)
    , angleSpan(angleSpan)
    , topDistorsionFactor(topDistorsionFactor)
    , bottomDistorsionFactor(bottomDistorsionFactor)
    , fisheyeAngle(fisheyeAngle)
{
}
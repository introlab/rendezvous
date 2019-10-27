#ifndef DEWAPING_CONFIG_H
#define DEWAPING_CONFIG_H

namespace Model
{
struct DewarpingConfig
{
    DewarpingConfig() = default;
    DewarpingConfig(float inRadius, float outRadius, float angleSpan, float topDistorsionFactor,
                    float bottomDistorsionFactor, float fisheyeAngle)
        : inRadius(inRadius)
        , outRadius(outRadius)
        , angleSpan(angleSpan)
        , topDistorsionFactor(topDistorsionFactor)
        , bottomDistorsionFactor(bottomDistorsionFactor)
        , fisheyeAngle(fisheyeAngle)
    {
    }

    float inRadius;
    float outRadius;
    float angleSpan;
    float topDistorsionFactor;
    float bottomDistorsionFactor;
    float fisheyeAngle;
};

}    // namespace Model

#endif    // !DEWAPING_CONFIG_H

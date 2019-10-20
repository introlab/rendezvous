#ifndef DEWAPING_CONFIG_H
#define DEWAPING_CONFIG_H

struct DewarpingConfig
{
    DewarpingConfig() = default;
    DewarpingConfig(float inRadius, float outRadius, float angleSpan, float topDistorsionFactor, float bottomDistorsionFactor, float fisheyeAngle);

    float inRadius;
    float outRadius;
    float angleSpan;
    float topDistorsionFactor;
    float bottomDistorsionFactor;
    float fisheyeAngle;

};

#endif // !DEWAPING_CONFIG_H

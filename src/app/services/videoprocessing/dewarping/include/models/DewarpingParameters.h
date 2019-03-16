#ifndef DEWAPING_PARAMETERS_H
#define DEWAPING_PARAMETERS_H

struct DewarpingParameters
{
public:

    DewarpingParameters();
    DewarpingParameters(float xCenter, float yCenter, float dewarpWidth, float dewarpHeight,
        float inRadius, float centerRadius, float outRadiusDiff, float xOffset, float bottomDistorsionFactor);

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

#endif // !DEWAPING_PARAMETERS_H


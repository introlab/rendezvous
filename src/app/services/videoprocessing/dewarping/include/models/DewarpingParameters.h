#ifndef DEWAPING_PARAMETERS_H
#define DEWAPING_PARAMETERS_H

struct DewarpingParameters
{
public:

    DewarpingParameters();
    DewarpingParameters(float xCenter, float yCenter, float dewarpWidth, float dewarpHeight, float inRadius, float centerRadius, 
        float outRadiusDiff, float xOffset, float bottomDistorsionFactor, float topOffset = 0, float bottomOffset = 0);

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

#endif // !DEWAPING_PARAMETERS_H


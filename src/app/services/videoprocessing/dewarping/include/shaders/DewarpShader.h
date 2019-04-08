#ifndef DEWARP_SHADER_H
#define DEWARP_SHADER_H

#include "ShaderProgram.h"

struct DewarpingParameters;

class DewarpShader : public ShaderProgram
{
public:

    DewarpShader();
    virtual ~DewarpShader();

    void loadDewarpingParameters(DewarpingParameters& dewarpingParameters);

private:

    void getAllUniformLocations();

private:

    GLint m_location_xCenter;
    GLint m_location_yCenter;
    GLint m_location_dewarpWidth;
    GLint m_location_dewarpHeight;
    GLint m_location_inRadius;
    GLint m_location_outRadiusDiff;
    GLint m_location_centerRadius;
    GLint m_location_xOffset;
    GLint m_location_bottomDistorsionFactor;
    GLint m_location_topOffset;
    GLint m_location_bottomOffset;

};

#endif // !DEWARP_SHADER_H


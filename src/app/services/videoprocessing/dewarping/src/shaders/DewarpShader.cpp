#include <shaders/DewarpShader.h>
#include <models/DewarpingParameters.h>
#include "autogen/dewarpVertexShader.cpp"
#include "autogen/dewarpFragmentShader.cpp"

DewarpShader::DewarpShader()
    : ShaderProgram(dewarpVertexShaderSource, dewarpFragmentShaderSource)
{
    getAllUniformLocations();
}

DewarpShader::~DewarpShader()
{
}

void DewarpShader::loadDewarpingParameters(DewarpingParameters& dewarpingParameters)
{
    loadFloat(m_location_xCenter, dewarpingParameters.xCenter);
    loadFloat(m_location_yCenter, dewarpingParameters.yCenter);
    loadFloat(m_location_dewarpWidth, dewarpingParameters.dewarpWidth);
    loadFloat(m_location_dewarpHeight, dewarpingParameters.dewarpHeight);
    loadFloat(m_location_inRadius, dewarpingParameters.inRadius);
    loadFloat(m_location_centerRadius, dewarpingParameters.centerRadius);
    loadFloat(m_location_outRadiusDiff, dewarpingParameters.outRadiusDiff);
    loadFloat(m_location_xOffset, dewarpingParameters.xOffset);
    loadFloat(m_location_bottomDistorsionFactor, dewarpingParameters.bottomDistorsionFactor);
    loadFloat(m_location_topOffset, dewarpingParameters.topOffset);
    loadFloat(m_location_bottomOffset, dewarpingParameters.bottomOffset);
}

void DewarpShader::getAllUniformLocations()
{
    m_location_xCenter = getUniformLocation("xCenter");
    m_location_yCenter = getUniformLocation("yCenter");
    m_location_dewarpWidth = getUniformLocation("dewarpWidth");
    m_location_dewarpHeight = getUniformLocation("dewarpHeight");
    m_location_inRadius = getUniformLocation("inRadius");
    m_location_centerRadius = getUniformLocation("centerRadius");
    m_location_outRadiusDiff = getUniformLocation("outRadiusDiff");
    m_location_xOffset = getUniformLocation("xOffset");
    m_location_bottomDistorsionFactor = getUniformLocation("bottomDistorsionFactor");
    m_location_topOffset = getUniformLocation("topOffset");
    m_location_bottomOffset = getUniformLocation("bottomOffset");
}
#include <shaders/DewarpShader.h>
#include "autogen/dewarpVertexShader.cpp"
#include "autogen/dewarpFragmentShader.cpp"

DewarpShader::DewarpShader()
    : ShaderProgram(dewarpVertexShaderSource, dewarpFragmentShaderSource)
{
}

DewarpShader::~DewarpShader()
{
}

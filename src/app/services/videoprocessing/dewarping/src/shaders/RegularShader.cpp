#include <shaders/RegularShader.h>
#include "autogen/regularVertexShader.cpp"
#include "autogen/regularFragmentShader.cpp"

RegularShader::RegularShader()
    : ShaderProgram(regularVertexShaderSource, regularFragmentShaderSource)
{
}

RegularShader::~RegularShader()
{
}

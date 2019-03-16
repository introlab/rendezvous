#include <shaders/ShaderProgram.h>
#include <models/DewarpingParameters.h>

#include <iostream>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

ShaderProgram::ShaderProgram(const char* vertexFileSource, const char * fragmentFileSource)
{
    m_vertexShaderID = loadShader(vertexFileSource, GL_VERTEX_SHADER);
    m_fragmentShaderID = loadShader(fragmentFileSource, GL_FRAGMENT_SHADER);
    initializeProgram();
    getAllUniformLocations();
}

ShaderProgram::~ShaderProgram()
{
}

void ShaderProgram::initializeProgram()
{
    m_programID = glCreateProgram();
    glAttachShader(m_programID, m_vertexShaderID);
    glAttachShader(m_programID, m_fragmentShaderID);
    bindAttribute(0, "position");
    glLinkProgram(m_programID);
    validateProgram();
}

GLint ShaderProgram::getUniformLocation(const GLchar * uniformName)
{
    return glGetUniformLocation(m_programID, uniformName);
}

void ShaderProgram::loadTransformation(glm::mat4& matrix)
{
    loadMatrix(m_location_transformationMatrix, matrix);
}

void ShaderProgram::loadDewarpingParameters(DewarpingParameters& dewarpingParameters)
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
}

void ShaderProgram::start()
{
    glUseProgram(m_programID);
}

void ShaderProgram::stop()
{
    glUseProgram(0);
}

void ShaderProgram::cleanUp()
{
    stop();
    glDetachShader(m_programID, m_vertexShaderID);
    glDetachShader(m_programID, m_fragmentShaderID);
    glDeleteShader(m_vertexShaderID);
    glDeleteShader(m_fragmentShaderID);
    glDeleteProgram(m_programID);
}

void ShaderProgram::bindAttribute(GLint attribute, const GLchar * variableName)
{
    glBindAttribLocation(m_programID, attribute, variableName);
}

void ShaderProgram::loadFloat(GLint location, GLfloat value)
{
    glUniform1f(location, value);
}

void ShaderProgram::loadMatrix(GLint location, glm::mat4& matrix)
{
    glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
    glm::value_ptr(matrix);
}

GLuint ShaderProgram::loadShader(const char * shaderSource, GLenum type)
{
    GLuint shaderID = glCreateShader(type);
    glShaderSource(shaderID, 1, &shaderSource, NULL);
    glCompileShader(shaderID);

    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        glGetShaderInfoLog(shaderID, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return shaderID;
}

void ShaderProgram::readShaderFile(const char * file, std::string& shaderCode)
{
    string line;
    ifstream myfile(file);

    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            shaderCode.append(line).append("\n");
        }
        myfile.close();
    }
    else
    {
        cout << "Unable to open file " << file << endl;
    }
}

void ShaderProgram::validateProgram()
{
    GLint success;
    GLchar infoLog[512];
    glGetProgramiv(m_programID, GL_LINK_STATUS, &success);

    if (!success) 
    {
        glGetProgramInfoLog(m_programID, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
}

void ShaderProgram::getAllUniformLocations()
{
    m_location_transformationMatrix = getUniformLocation("transformationMatrix");
    m_location_xCenter = getUniformLocation("xCenter");
    m_location_yCenter = getUniformLocation("yCenter");
    m_location_dewarpWidth = getUniformLocation("dewarpWidth");
    m_location_dewarpHeight = getUniformLocation("dewarpHeight");
    m_location_inRadius = getUniformLocation("inRadius");
    m_location_centerRadius = getUniformLocation("centerRadius");
    m_location_outRadiusDiff = getUniformLocation("outRadiusDiff");
    m_location_xOffset = getUniformLocation("xOffset");
    m_location_bottomDistorsionFactor = getUniformLocation("bottomDistorsionFactor");
}

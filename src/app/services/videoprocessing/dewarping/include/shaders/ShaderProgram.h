#ifndef SHADER_PROGRAM_H
#define SHADER_PROGRAM_H

#include <string>
#include <glm/glm.hpp>
#include <glad/glad.h>

struct DewarpingParameters;

class ShaderProgram
{
public:

    ShaderProgram(const char * vertexFileSource, const char * fragmentFileSource);
    virtual ~ShaderProgram();
    
    GLint getUniformLocation(const GLchar * uniformName);
    void loadTransformation(glm::mat4& matrix);
    void loadDewarpingParameters(DewarpingParameters& dewarpingParameters);
    void start();
    void stop();
    void cleanUp();

protected:

    void initializeProgram();
    void bindAttribute(GLint attribute, const GLchar * variableName);
    void loadFloat(GLint m_location, GLfloat value);
    void loadMatrix(GLint m_location, glm::mat4& matrix);
    GLuint loadShader(const char * shaderSource, GLenum type);
    void readShaderFile(const char * file, std::string& shaderCode);
    void validateProgram();
    void getAllUniformLocations();

private:

    GLuint m_programID;
    GLuint m_vertexShaderID;
    GLuint m_fragmentShaderID;

    GLint m_location_transformationMatrix;
    GLint m_location_xCenter;
    GLint m_location_yCenter;
    GLint m_location_dewarpWidth;
    GLint m_location_dewarpHeight;
    GLint m_location_inRadius;
    GLint m_location_outRadiusDiff;
    GLint m_location_centerRadius;
    GLint m_location_xOffset;
    GLint m_location_bottomDistorsionFactor;
};

#endif // !SHADER_PROGRAM_H

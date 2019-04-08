#ifndef SHADER_PROGRAM_H
#define SHADER_PROGRAM_H

#include <string>
#include <glm/glm.hpp>
#include <glad/glad.h>

class ShaderProgram
{
public:

    ShaderProgram(const char * vertexFileSource, const char * fragmentFileSource);
    virtual ~ShaderProgram();
    
    GLint getUniformLocation(const GLchar * uniformName);
    void loadTransformation(glm::mat4& matrix);
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
    
};

#endif // !SHADER_PROGRAM_H
